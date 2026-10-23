"""Independent numerical replay and declared-domain audit of saved study models."""
import argparse
from pathlib import Path
import numpy as np
from dimensr import DimenSRRegressor, SearchConfig
from dimensr.domains import domain_probes
from dimensr.intervals import audit_box
import interval_search_study as study


def raw_predict(tree, constants, X):
    """Replay serialized operators directly; keep every intermediate finite mask."""
    values = iter(constants)
    valid = np.ones(len(X), dtype=bool)
    binary = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide}
    unary = {"neg": np.negative, "square": np.square, "sqrt": np.sqrt,
             "sin": np.sin, "cos": np.cos, "log": np.log, "exp": np.exp}

    def visit(node):
        token, children = node["token"], node["children"]
        if not children:
            result = X[:, int(token[1:])] if token.startswith("x") else next(values)
        elif token in binary:
            result = binary[token](visit(children[0]), visit(children[1]))
        else:
            result = unary[token](visit(children[0]))
        valid[:] &= np.isfinite(result)
        return result

    with np.errstate(all="ignore"):
        output = np.broadcast_to(visit(tree), (len(X),)).astype(float, copy=True)
    output[~valid] = np.nan
    return output


def verify(out):
    study.score(out)  # Recompute metrics, budgets, unit checks, and evidence hashes.
    protocol = study.checked(out)
    models = candidates = declared_domain_checks = 0
    final_box_audits = []
    for job in protocol["jobs"]:
        run = study.read(out / "runs" / f"{job['id']}.json")
        if run["status"] != "ok":
            continue
        path = out / "models" / f"{job['id']}.json"
        artifact = study.read(path)
        model = DimenSRRegressor.load(path)
        models += 1
        final_box_audits.append(dict(id=job["id"], variant=job["variant"],
                                    **model.audit_domain(protocol["cases"][job["case"]]["interval_bounds"], max_boxes=32)))
        with np.load(out / "data" / f"{job['data_key']}_fit.npz") as fit:
            if job["variant"] != "v03":
                probes, audit = domain_probes(fit["X_train"], SearchConfig(**run["config"]), X_domain=fit["X_domain"], input_bounds=protocol["cases"][job["case"]]["interval_bounds"])
                assert audit == model.diagnostics_["domain_audit"]
            for split in ("train", "val"):
                predicted = raw_predict(artifact["best"]["tree"], artifact["best"]["constants"], fit[f"X_{split}"])
                np.testing.assert_array_equal(predicted, model.predict(fit[f"X_{split}"]))
                rmse = np.sqrt(np.mean((predicted - fit[f"y_{split}"])**2))
                expected = artifact["best"]["train_rmse"] if split == "train" else artifact["validation_rmse"]
                np.testing.assert_allclose(rmse, expected, rtol=1e-9, atol=1e-12)
            with np.load(out / "data" / f"{job['data_key']}_holdout.npz") as holdout:
                from dimensr.expression import Node
                for candidate in [artifact["best"]] + artifact["pareto_front"]:
                    candidates += 1
                    tree = Node.from_dict(candidate["tree"])
                    for split in ("test", "ood", "boundary", "stress"):
                        X = holdout[f"X_{split}"]
                        expected = raw_predict(candidate["tree"], candidate["constants"], X)
                        np.testing.assert_array_equal(tree.evaluate(X, candidate["constants"]), expected)
                    if job["variant"] != "v03":
                        assert np.isfinite(raw_predict(candidate["tree"], candidate["constants"], probes)).all()
                        declared_domain_checks += 1
                        if job["variant"] == "v04":
                            assert audit_box(tree, candidate["constants"], protocol["cases"][job["case"]]["interval_bounds"], max_boxes=run["config"].get("interval_max_boxes", 32))["status"] == "enclosed"
    result = dict(status="verified", models=models, saved_trees_including_best=candidates,
                  declared_domain_checks=declared_domain_checks,
                  final_box_audits=final_box_audits,
                  verifier_sha256=study.sha(__file__), protocol_sha256=study.sha(out / "protocol.json"),
                  scores_sha256=study.sha(out / "scores.json"),
                  scope="Independent raw-tree replay, saved-model predictions, training/validation metrics, candidate units/budgets, and new-variant declared probes; no continuous-domain proof")
    study.write(out / "verification.json", result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, default=study.ROOT / "results/interval_search_v4")
    args = parser.parse_args()
    verify(args.out.resolve())
