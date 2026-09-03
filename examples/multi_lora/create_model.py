from peft import LoraConfig

# Where to save the LoRA configuration
# OUTPUT_PATH = "/path/to/lora_config"
OUTPUT_PATH = "/home/tim/models/Qwen3.8-27B-lora-r16-new"

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

config.save_pretrained(OUTPUT_PATH)

print(f"LoRA config saved to: {OUTPUT_PATH}")