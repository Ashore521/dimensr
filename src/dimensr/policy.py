"""GRU token policy: on-policy risk-seeking gradient + separate GP imitation."""
import numpy as np
import torch
from torch import nn


class Policy(nn.Module):
    def __init__(self, n_tokens, hidden_size=32, learning_rate=0.003, seed=0):
        super().__init__()
        self.n_tokens = n_tokens
        self.bos = n_tokens
        # Local initialization and generator avoid resetting the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(n_tokens + 1, 12)
            self.rnn = nn.GRU(33, hidden_size, batch_first=True)
            self.head = nn.Linear(hidden_size, n_tokens)
            self.production_query = nn.Linear(hidden_size + 12, 16)
        self.generator = torch.Generator().manual_seed(seed)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, previous, parents, contexts, hidden=None, return_states=False):
        inputs = torch.cat((self.embedding(previous), self.embedding(parents), contexts), dim=-1)
        output, hidden = self.rnn(inputs, hidden)
        result = (self.head(output), hidden)
        return result + (output,) if return_states else result

    @torch.no_grad()
    def sample_step(self, previous, parent, context, mask, hidden):
        logits, hidden = self(torch.tensor([[previous]]), torch.tensor([[parent]]),
                              torch.tensor([[context]], dtype=torch.float32), hidden)
        logits = logits[0, 0].masked_fill(~torch.as_tensor(mask), -1e9)
        token = torch.multinomial(torch.softmax(logits, -1), 1, generator=self.generator)
        return int(token.item()), hidden

    @torch.no_grad()
    def sample_production(self, hidden, features, token):
        state = torch.cat((hidden[-1, 0], self.embedding(torch.tensor(token))))
        logits = torch.as_tensor(features) @ self.production_query(state) / 4
        return int(torch.multinomial(torch.softmax(logits, -1), 1, generator=self.generator).item())

    def log_probs(self, traces):
        batch = len(traces)
        length = max(len(t.tokens) for t in traces)
        tokens = torch.zeros((batch, length), dtype=torch.long)
        previous = torch.full_like(tokens, self.bos)
        parents = torch.full_like(tokens, self.bos)
        contexts = torch.zeros((batch, length, 9))
        masks = torch.ones((batch, length, self.n_tokens), dtype=torch.bool)
        active = torch.zeros((batch, length))
        for i, trace in enumerate(traces):
            n = len(trace.tokens)
            tokens[i, :n] = torch.tensor(trace.tokens)
            previous[i, 1:n] = torch.tensor(trace.tokens[:-1])
            parents[i, :n] = torch.tensor(trace.parents)
            contexts[i, :n] = torch.tensor(trace.contexts)
            masks[i, :n] = torch.as_tensor(np.asarray(trace.masks))
            active[i, :n] = 1
        logits, _, states = self(previous, parents, contexts, return_states=True)
        logp = torch.log_softmax(logits.masked_fill(~masks, -1e9), -1)
        selected = logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        entropy = -(logp.exp() * logp).sum(-1)
        if any(trace.production_features for trace in traces):
            max_choices = max(len(f) for t in traces for f in t.production_features)
            features = torch.zeros((batch, length, max_choices, 16))
            legal = torch.zeros((batch, length, max_choices), dtype=torch.bool)
            legal[:, :, 0] = True
            chosen = torch.zeros((batch, length), dtype=torch.long)
            for i, trace in enumerate(traces):
                for j, f in enumerate(trace.production_features):
                    features[i, j, :len(f)] = torch.as_tensor(f)
                    legal[i, j, :len(f)] = True
                    chosen[i, j] = trace.production_indices[j]
            query = self.production_query(torch.cat((states, self.embedding(tokens)), dim=-1))
            split_logits = (features * query.unsqueeze(2)).sum(-1) / 4
            split_logp = torch.log_softmax(split_logits.masked_fill(~legal, -1e9), -1)
            selected = selected + split_logp.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
            entropy = entropy - (split_logp.exp() * split_logp).sum(-1)
        return (selected * active).sum(-1), (entropy * active).sum(-1)

    def update(self, traces, rewards, elite_traces=(), risk_fraction=0.3,
               entropy_weight=0.01, imitation_weight=0.1):
        logp, entropy = self.log_probs(traces)
        r = torch.tensor(rewards, dtype=torch.float32)
        baseline = torch.quantile(r, 1 - risk_fraction)
        advantage = torch.where(r >= baseline, r - baseline, 0.0)
        pg = -(advantage * logp).mean() / risk_fraction
        loss = pg - entropy_weight * entropy.mean()
        imitation = torch.tensor(0.0)
        if elite_traces and imitation_weight:
            elite_logp, _ = self.log_probs(elite_traces)
            lengths = torch.tensor([len(t.tokens) for t in elite_traces])
            imitation = -(elite_logp / lengths).mean()
            loss = loss + imitation_weight * imitation
        self.optimizer.zero_grad()
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return {"policy_loss": float(loss.detach()), "policy_gradient": float(pg.detach()),
                "imitation_loss": float(imitation.detach()), "gradient_norm": float(norm),
                "reward_threshold": float(baseline)}
