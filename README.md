# Training Methods
* follow [transformers notebook](https://colab.research.google.com/drive/13r94i6Fh4oYf-eJRSi7S_y_cen5NYkBm?usp=sharing)
* epoch = 10
## Apply LoRA
### v1 Config
```python=
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=["q_lin", "v_lin", "k_lin", "out_lin", "ffn.lin1", "ffn.lin2"]
)
```
### v1 Result
![image](https://i.imgur.com/kQsSJv9.png)
TrainOutput(global_step=80, training_loss=0.6802545070648194, metrics={'train_runtime': 14.4446, 'train_samples_per_second': 88.615, 'train_steps_per_second': 5.538, 'train_loss': 0.6802545070648194, 'epoch': 10.0})

### v2 Config
set rank=32 on v1 config
```python=
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=32,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=["q_lin", "v_lin", "k_lin", "out_lin", "ffn.lin1", "ffn.lin2"]
)
```
### v2 Result
![image](https://i.imgur.com/NLcXMGp.png)
TrainOutput(global_step=80, training_loss=0.6876016139984131, metrics={'train_runtime': 14.2093, 'train_samples_per_second': 90.082, 'train_steps_per_second': 5.63, 'train_loss': 0.6876016139984131, 'epoch': 10.0})

### v3 Config
set use_rslora=True on v1 config
```python=
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=["q_lin", "v_lin", "k_lin", "out_lin", "ffn.lin1", "ffn.lin2"],
    use_rslora = True
)
```
### v3 Result
![image](https://i.imgur.com/LAkEy8k.png)
TrainOutput(global_step=80, training_loss=0.6797160625457763, metrics={'train_runtime': 14.5809, 'train_samples_per_second': 87.786, 'train_steps_per_second': 5.487, 'train_loss': 0.6797160625457763, 'epoch': 10.0})

## Apply IA3
### Config
```python=
peft_config = IA3Config(
    target_modules=["q_lin", "v_lin", "k_lin", "out_lin", "ffn.lin1", "ffn.lin2"],
    task_type="SEQ_CLS",
    feedforward_modules=["ffn.lin1", "ffn.lin2"]
)
```
### Result
![image](https://i.imgur.com/bXyifx6.png)
TrainOutput(global_step=80, training_loss=0.6880824089050293, metrics={'train_runtime': 11.448, 'train_samples_per_second': 111.81, 'train_steps_per_second': 6.988, 'train_loss': 0.6880824089050293, 'epoch': 10.0})
