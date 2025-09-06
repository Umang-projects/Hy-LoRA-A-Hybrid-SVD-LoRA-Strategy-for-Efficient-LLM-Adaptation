# -*- coding: utf-8 -*-
#Experiment: Enhanced Training Approach (Hyperparameter Grid Search)
if __name__ == "__main__":
    # Hyperparameter Grid
    ALPHAS = [0.5, 0.7]
    LORA_RANKS = [16, 32]

    # Fixed Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Training Config
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 4
    MAX_LENGTH = 256
    NUM_EPOCHS = 3
    NUM_TRAIN_SAMPLES = 2000

    # --- Load Data (One-time operation) ---
    print("Loading datasets…")
    dolly_ds = load_dataset("databricks/databricks-dolly-15k", split="train", trust_remote_code=True)
    wikitext_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)

    train_texts = [
        f"Instruction:\n{ex['instruction']}\n\nContext:\n{ex['context']}\n\nResponse:\n{ex['response']}"
        for ex in dolly_ds.select(range(NUM_TRAIN_SAMPLES))
    ]
    eval_texts = [t for t in wikitext_ds["text"][:100] if t and t.strip()]

    # --- Load Original Model (One-time) ---
    print(f"\nLoading base model {MODEL_ID}…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    orig_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE)

    # Precompute baseline perplexity
    print("\nComputing baseline perplexity…")
    base_ppl = calculate_perplexity(orig_model, tokenizer, eval_texts, DEVICE, max_length=MAX_LENGTH)
    print(f"Baseline Perplexity: {base_ppl:.2f}\n")

    # Results storage
    results = []

    # --- Hyperparameter Grid Search ---
    for alpha in ALPHAS:
        for lora_rank in LORA_RANKS:
            print(f"\n{'='*50}")
            print(f"▶ Experiment: α={alpha} | LoRA Rank={lora_rank}")
            print(f"{'='*50}")

            # --- STAGE 1: SVD Compression ---
            comp_model = copy.deepcopy(orig_model)
            num_replaced = replace_with_svd_lora(comp_model, alpha, lora_rank, torch.bfloat16)
            comp_model = setup_model_for_lora_training(comp_model)

            # Pre-tune evaluation
            pre_tune_ppl = calculate_perplexity(comp_model, tokenizer, eval_texts, DEVICE, max_length=MAX_LENGTH)
            print(f"Pre-tune PPL: {pre_tune_ppl:.2f}")

            # --- STAGE 2: Fine-tuning ---
            comp_model = train_lora_adapters(
                comp_model,
                tokenizer,
                train_texts,
                DEVICE,
                epochs=NUM_EPOCHS,
                lr=LEARNING_RATE,
                batch_size=BATCH_SIZE,
                max_length=MAX_LENGTH
            )

            # Post-tune evaluation
            post_tune_ppl = calculate_perplexity(comp_model, tokenizer, eval_texts, DEVICE, max_length=MAX_LENGTH)
            print(f"Post-tune PPL: {post_tune_ppl:.2f}")

            # Parameter stats
            orig_params = sum(p.numel() for p in orig_model.parameters())
            comp_params = sum(p.numel() for p in comp_model.parameters())
            reduction = (1 - comp_params/orig_params) * 100

            # Store results
            results.append({
                "alpha": alpha,
                "lora_rank": lora_rank,
                "pre_tune_ppl": pre_tune_ppl,
                "post_tune_ppl": post_tune_ppl,
                "param_reduction": reduction
            })

            # Clean up to save memory
            del comp_model
            torch.cuda.empty_cache()

    # --- Final Report ---
    print("\n\n" + "="*60)
    print("FINAL EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    print(f"Base Model PPL: {base_ppl:.2f}")
    print(f"Training Samples: {NUM_TRAIN_SAMPLES}")
    print("-"*60)
    print(f"{'α':<5}{'LoRA':<6}{'Pre-Tune':<10}{'Post-Tune':<10}{'Reduction':<10}")
    print("-"*60)

    for res in results:
        print(f"{res['alpha']:<5.1f}{res['lora_rank']:<6}{res['pre_tune_ppl']:<10.2f}{res['post_tune_ppl']:<10.2f}{res['param_reduction']:<10.2f}%")
    print("="*60)
