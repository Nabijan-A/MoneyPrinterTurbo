# LLM Fine-tuning Practical Case: Optimizing Video Script Generation

## Overview

This document provides a practical case study of LLM (Large Language Model) fine-tuning, demonstrating how to fine-tune language models specifically for MoneyPrinterTurbo's video script generation tasks to improve content quality and relevance.

## Why Fine-tune?

While general-purpose models (like GPT-3.5, GPT-4, DeepSeek, etc.) already have strong content generation capabilities, fine-tuning for specific domains or styles offers several advantages:

1. **Better Style Matching**: Fine-tuned models better align with your content style and tone
2. **Improved Content Quality**: Optimized specifically for video script generation
3. **Lower Generation Costs**: Achieve the same results with smaller, cheaper models
4. **Faster Generation**: Smaller models typically respond faster
5. **Reduced Prompt Engineering**: Models understand your needs without complex prompts

## Practical Case: Fine-tuning GPT-3.5 for Science Education Videos

### Scenario Description

Suppose you primarily create "science education" short videos and want to fine-tune a specialized model to generate scripts for this type of content.

### Step 1: Prepare Training Data

Fine-tuning requires high-quality training data. For video script generation tasks, we need to prepare "input-output" pairs:

**Data Format (JSONL):**

```jsonl
{"messages": [{"role": "system", "content": "You are a professional science education video scriptwriter, skilled at transforming complex scientific knowledge into simple, engaging short video scripts."}, {"role": "user", "content": "Topic: Why is the sky blue?"}, {"role": "assistant", "content": "Have you ever wondered why the sky we see is always blue?\n\nThe answer lies in the nature of light. Sunlight appears white, but it's actually composed of seven colors: red, orange, yellow, green, blue, indigo, and violet.\n\nWhen sunlight enters Earth's atmosphere, it encounters various tiny particles in the air. These particles scatter light in all directions. The key point is that blue light has a shorter wavelength and is more easily scattered.\n\nJust like in a crowd, shorter people are more easily blocked, short-wavelength blue light is more easily 'blocked' by air particles and scattered into our eyes.\n\nSo when we look up at the sky, what we mainly see is this scattered blue light. That's why the sky appears beautiful blue.\n\nInterestingly, during sunrise and sunset, sunlight passes through a thicker atmosphere. Blue light is almost completely scattered away, leaving only red and orange light, which is why the sky turns red or orange. Nature is truly a magical colorist!"}]}
{"messages": [{"role": "system", "content": "You are a professional science education video scriptwriter, skilled at transforming complex scientific knowledge into simple, engaging short video scripts."}, {"role": "user", "content": "Topic: How are black holes formed?"}, {"role": "assistant", "content": "Black holes, one of the most mysterious objects in the universe - how are they born?\n\nIt all starts with stars. When a massive star reaches the end of its life, its core fuel is depleted and can no longer generate enough pressure to counteract its own gravity.\n\nAt this point, the star undergoes a violent collapse. Imagine a celestial body dozens of times larger than the Sun contracting to an extremely small point in just a few seconds - the energy released is tremendous!\n\nDuring the collapse, the star's outer layers are blown away, forming a supernova explosion, while the core continues to contract. If the star's mass exceeds 25 times that of the Sun, the core keeps contracting until it forms a singularity with infinite density and infinitely small volume.\n\nThe gravity around this singularity is so strong that even light cannot escape, forming a black hole. There's a boundary around the black hole called the 'event horizon' - once you cross this boundary, nothing can escape, including light.\n\nSo, black holes are actually the most extreme gravitational traps in the universe, the 'legacy' left behind after massive stars die. Although we can't see them, we can detect their presence through their effects on surrounding matter."}]}
{"messages": [{"role": "system", "content": "You are a professional science education video scriptwriter, skilled at transforming complex scientific knowledge into simple, engaging short video scripts."}, {"role": "user", "content": "Topic: Why do we dream?"}, {"role": "assistant", "content": "Everyone dreams, but do you know why?\n\nScientists have discovered through years of research that dreaming mainly occurs during the Rapid Eye Movement sleep stage, commonly known as REM sleep. During this stage, our brain is exceptionally active, almost as active as when we're awake.\n\nSo why does the brain stay so active during sleep? There are several mainstream theories.\n\nThe first theory suggests that dreaming is the brain organizing and consolidating memories. Like a computer organizing files in the background, the brain categorizes and stores daytime experiences during sleep, keeping important ones and deleting unimportant ones.\n\nThe second theory proposes that dreaming is the brain regulating emotions. Through dreams, we can process daytime stress and emotions, which is why many people repeatedly dream about related content after experiencing major events.\n\nAnother interesting theory suggests that dreaming is the brain 'rehearsing.' By simulating various scenarios, it helps us better cope with challenges and dangers in reality.\n\nAlthough we can't fully explain all the mysteries of dreaming, we can confirm that dreaming is important for our mental health and cognitive function. So next time you have a strange dream, don't worry - it's just your brain working hard!"}]}
```

**Data Preparation Recommendations:**

1. **Quantity**: Prepare at least 50-100 high-quality examples, more is better (recommended 200-500)
2. **Quality**: Ensure each example is a script you're satisfied with, as the model will learn these styles
3. **Diversity**: Cover different topics and perspectives to avoid overly uniform patterns
4. **Consistent Format**: Maintain consistency in system prompts, user inputs, and output formats

### Step 2: Fine-tune Using OpenAI API

#### 2.1 Install OpenAI Python Library

```bash
pip install openai
```

#### 2.2 Prepare Fine-tuning Script

Create a Python script `finetune_model.py`:

```python
import openai
import json
import time

# Set your OpenAI API Key
openai.api_key = "your-api-key-here"

# 1. Upload training data file
print("Uploading training data...")
with open("training_data.jsonl", "rb") as f:
    training_file = openai.File.create(
        file=f,
        purpose="fine-tune"
    )

training_file_id = training_file.id
print(f"Training file uploaded, ID: {training_file_id}")

# Wait for file processing
print("Waiting for file processing...")
while True:
    file_status = openai.File.retrieve(training_file_id)
    if file_status.status == "processed":
        print("File processing complete!")
        break
    elif file_status.status == "error":
        print("File processing error!")
        exit(1)
    time.sleep(10)

# 2. Create fine-tuning job
print("Creating fine-tuning job...")
fine_tune_job = openai.FineTuningJob.create(
    training_file=training_file_id,
    model="gpt-3.5-turbo",  # Base model
    hyperparameters={
        "n_epochs": 3,  # Training epochs, adjust as needed
    }
)

job_id = fine_tune_job.id
print(f"Fine-tuning job created, ID: {job_id}")

# 3. Monitor fine-tuning progress
print("Starting fine-tuning, this may take minutes to hours...")
while True:
    job_status = openai.FineTuningJob.retrieve(job_id)
    status = job_status.status
    
    print(f"Current status: {status}")
    
    if status == "succeeded":
        print("Fine-tuning successful!")
        fine_tuned_model = job_status.fine_tuned_model
        print(f"Fine-tuned model ID: {fine_tuned_model}")
        break
    elif status in ["failed", "cancelled"]:
        print(f"Fine-tuning failed or cancelled: {status}")
        break
    
    time.sleep(60)  # Check every minute

print("Fine-tuning complete!")
```

#### 2.3 Run Fine-tuning

```bash
python finetune_model.py
```

The fine-tuning process may take minutes to hours, depending on data size and queue.

### Step 3: Use Fine-tuned Model in MoneyPrinterTurbo

After fine-tuning completes, you'll get a new model ID in the format: `ft:gpt-3.5-turbo-0613:your-org::7p4lURel`

Configure in `config.toml`:

```toml
llm_provider = "openai"
openai_api_key = "your-api-key"
openai_model_name = "ft:gpt-3.5-turbo-0613:your-org::7p4lURel"  # Use fine-tuned model
```

### Step 4: Test and Evaluate

Create a test script to compare before and after fine-tuning:

```python
import openai

openai.api_key = "your-api-key"

test_topic = "Why does it rain?"

# Test original model
print("=== Original Model ===")
response_original = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a professional science education video scriptwriter."},
        {"role": "user", "content": f"Topic: {test_topic}"}
    ]
)
print(response_original.choices[0].message.content)

# Test fine-tuned model
print("\n=== Fine-tuned Model ===")
response_finetuned = openai.ChatCompletion.create(
    model="ft:gpt-3.5-turbo-0613:your-org::7p4lURel",
    messages=[
        {"role": "system", "content": "You are a professional science education video scriptwriter."},
        {"role": "user", "content": f"Topic: {test_topic}"}
    ]
)
print(response_finetuned.choices[0].message.content)
```

## Fine-tuning Options for Other Platforms

### DeepSeek Fine-tuning

DeepSeek also supports model fine-tuning with lower costs, especially user-friendly for Chinese users.

Visit [DeepSeek Platform](https://platform.deepseek.com/) for fine-tuning details.

### Moonshot Fine-tuning

Visit [Moonshot Console](https://platform.moonshot.cn/) to explore fine-tuning options.

### Open-Source Model Fine-tuning (More Flexible but Requires More Technical Skills)

If you want complete control over the fine-tuning process, you can use open-source models like Llama, Qwen:

1. **Choose Base Model**: Such as Llama-2-7B, Qwen-7B
2. **Use Fine-tuning Framework**: Such as [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
3. **Prepare Training Environment**: Requires GPU (recommend at least RTX 3090 or A100)
4. **Execute Fine-tuning**: Using LoRA or QLoRA techniques can significantly reduce memory requirements

Example using LLaMA-Factory:

```bash
# Install LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -r requirements.txt

# Prepare data (convert to LLaMA-Factory format)
# Run fine-tuning
python src/train_bash.py \
    --model_name_or_path Qwen/Qwen-7B \
    --do_train \
    --dataset your_dataset \
    --finetuning_type lora \
    --output_dir output \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3
```

## Fine-tuning Cost Estimation

### OpenAI GPT-3.5 Turbo Fine-tuning Costs (as of 2024)

- **Training Cost**: ~$0.008 / 1K tokens
- **Usage Cost**: ~$0.012 / 1K tokens (input) + $0.016 / 1K tokens (output)

Example:
- 200 training samples, average 500 tokens each = 100K tokens
- Training cost: 100 × $0.008 = $0.80
- Training 3 epochs: $0.80 × 3 = $2.40

### DeepSeek Fine-tuning Costs

- Typically 50%-70% cheaper than OpenAI
- Visit their website for specific pricing

## Best Practice Recommendations

1. **Start Small**: Begin with 50-100 samples to validate the approach before scaling
2. **Maintain Data Quality**: Better to have fewer high-quality samples than many poor ones
3. **Iterate and Optimize**: Continuously adjust training data and parameters based on test results
4. **Version Control**: Save data and models from each fine-tuning run for comparison and rollback
5. **Cost Management**: Validate on smaller models first, then consider larger models if needed
6. **Regular Updates**: Regularly fine-tune with new data as your content style evolves

## Evaluating Fine-tuning Results

Evaluate fine-tuned models across these dimensions:

1. **Content Relevance**: Does generated content match the topic?
2. **Style Consistency**: Does it match your expected tone and style?
3. **Structure Rationality**: Is the script structure suitable for video presentation?
4. **Creativity Level**: Does it have sufficient innovation and appeal?
5. **Appropriate Length**: Is the generated script length suitable?

Prepare a test set (10-20 topics), compare generation results before and after fine-tuning, and score them.

## Important Considerations

1. **Data Privacy**: Ensure training data doesn't contain sensitive information
2. **Copyright Compliance**: Training data should be your own creation or content you have rights to use
3. **Overfitting Risk**: If training data is too uniform, the model may lose flexibility
4. **Regular Evaluation**: Fine-tuned models may become outdated over time; regular evaluation and updates are needed

## Summary

Through targeted fine-tuning, you can make large language models better understand and generate video scripts that meet your needs. While fine-tuning requires time and cost investment, in the long run, it can significantly improve content quality and production efficiency.

Start with small-scale experiments, gradually optimize, and find the fine-tuning solution that works best for you.

## Reference Resources

- [OpenAI Fine-tuning Documentation](https://platform.openai.com/docs/guides/fine-tuning)
- [DeepSeek Fine-tuning Guide](https://platform.deepseek.com/docs)
- [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)
- [Hugging Face Fine-tuning Tutorial](https://huggingface.co/docs/transformers/training)

---

*Last updated: October 2025*
