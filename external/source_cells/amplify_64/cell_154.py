# Chạy lại benchmark nhiều seed
SEEDS = [42, 52, 62, 72, 82]

all_results = []

for seed in SEEDS:
    print("\n" + "=" * 70)
    print("RUNNING SEED:", seed)
    print("=" * 70)
    set_seed(seed)

    model_dict = {
        "seq_only": SeqOnlyModel(seq_dim=seq_dim),
        "phys_only": PhysOnlyModel(phys_dim=phys_dim),
        "concat": FusionConcatModel(seq_dim=seq_dim, phys_dim=phys_dim),
        "cross_attn": CrossModalModel(seq_dim=seq_dim, phys_dim=phys_dim),
    }

    for name, model in model_dict.items():
        print(f"\n----- {name} | seed={seed} -----")
        trained = train_one_model(model, train_ds, val_ds, epochs=12, lr=1e-3, batch_size=64, patience=4)
        y_true, y_prob = predict_model(trained, test_ds, batch_size=128)
        metrics = binary_metrics(y_true, y_prob, thr=0.5)
        metrics["model"] = name
        metrics["seed"] = seed
        all_results.append(metrics)

results_df = pd.DataFrame(all_results)
results_df