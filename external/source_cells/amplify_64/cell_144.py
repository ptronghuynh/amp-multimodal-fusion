# CELL 7 — Smoke test 1 seed cho 4 model
set_seed(42)

models = {
    "seq_only": SeqOnlyModel(seq_dim=seq_dim),
    "phys_only": PhysOnlyModel(phys_dim=phys_dim),
    "concat": FusionConcatModel(seq_dim=seq_dim, phys_dim=phys_dim),
    "cross_attn": CrossModalModel(seq_dim=seq_dim, phys_dim=phys_dim),
}

smoke_results = []

for name, model in models.items():
    print(f"\n========== {name} ==========")
    trained = train_one_model(model, train_ds, val_ds, epochs=12, lr=1e-3, batch_size=64, patience=4)
    y_true, y_prob = predict_model(trained, test_ds, batch_size=128)
    metrics = binary_metrics(y_true, y_prob, thr=0.5)
    metrics["model"] = name
    metrics["seed"] = 42
    smoke_results.append(metrics)

    ckpt_path = OUT_DIR / f"{name}_seed42.pt"
    torch.save(trained.state_dict(), ckpt_path)
    print("Saved:", ckpt_path)
    print(metrics)

smoke_df = pd.DataFrame(smoke_results)
smoke_df