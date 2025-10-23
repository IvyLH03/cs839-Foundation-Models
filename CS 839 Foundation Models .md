# CS 839 Foundation Models 

Fred Sala

Office Hour: Thurs 2:30 - 4:00 PM in Morgridge 5514

### Supervised Learning

**Problem Setting**

- Set of possible instances: $X$
- Unknown target function: $f: X -> Y$
- Set of models (a.k.a. hypotheses): $H = \{h |h: X -> Y\}$

**Get**

- Training set of instances for unknown target function
- $(xˆ{(1)}, yˆ{(1)}), ...$

#### Input space, output space, hypothesis class

**Output spcae**: Classification/regression

- Discrete: classification
- Continuous: regression
- Other types: 

**Hypothesis class**

- Ex. linear models: $h_\theta (x) = \theta_0 +\theta_1x_1 + ...$
- feedforward neural networks
- Parameters: weights/thetas

**Goal**: model h that best approximates f

- One way: empirical risk minimization (ERM)
- Generalization? new points

**Evaluation**

- **Validation set** (tuning set): not used for primary training process, used to select among models
- **Test set**: Not used for training or selection, compute metrics

**Overfitting**

- Low error on the training data, high error on the entire distribution
- Doesn't need to happen on this class
- Reduce model complexicity because it may try to hard to fit the training data

### Neural networks

**Perceptron**: simple network

**Neural Networks: Multilayer Perceptrons**

**Training**

- Compute gradient, update weights

**Convolution Layers**

X input matrix, W kernel matrix (small window/filter), b bias (a scalar), Y output matrix

**CNN**

- **Input**: volume channels * height * width
- Hyperparameters
- Output
- Parameters

**CNN Architecture: AlexNet**





**Self Supervision**

- **Pretext tasks**: "Fake tasks" that can be put on unlabeled dataset. E.g. image completion for classification
- Use the learned network as a feature extractor for supervised learning (on small amount of labeled data)



**Generative vs. Discriminative**

Generative: learning a distribution from samples

Embedddings & Representations



# Architecture

### Attention

Motivation: fixed context vector not enough. Need to make model focus on the right part. 

#### **Self-attention**

https://jalammar.github.io/illustrated-transformer/

Focus within sentence

**Query, key, value**

- Query = WˆQ * Embedding, Key = WˆK * Embedding, Value = WˆK * Embedding
- **Score function**: compare query and key to get a score
  - Simpler is usually better (e.g. dot product, q1*k2)
  - A word query with the key of all other words in the sequence
  - Then do a softmax (add up to 1, kind of "probability"). There might be negatives in the dot product results, but it will go to almost 0 after softmax. 

**Multi-head**: do the same thing multiple times in parallel, with different weights, to capture different patterns

>  ==How can we end up with multiple different weights during training?==
>
> Different random initialization? So they will look to different directions? 

##### **Positional Encodings**

**Sinusoidal embeddings**

$PE_{(pos, 2i)} = sin(pos/10000ˆ{2i/d_{model}})$

$PE_{(pos, 2i+1)} = cos(pos/10000ˆ{2i/d_{model}})$

**Position vs. Location index**

- Position ($pos$): position in the sentence. e.g. 13th word in the sentence
- Location index ($i$): location in the vector

Apply the PE to the **embedding**, and then go to the attention

We want the PE to be consistent (no matter what the input is), smooth (no spikes), Linearity across positions

**Modern Positional Encodings**

- Rotary Positional Encoding

> Position embedding is designed to consider relative positions. for sinusoidal attention, it's absolute position, but with the property of sin/cos, x+k and y+k considered relative position. for rotary positional encoding, it's actually relative(?). Positional embeddings are designed to be smooth and continuous that if you shift the sentence by 1 word, it's not going to change much.

### Model Architecture

#### Sequence-sequence model (e.g. translation)

- Multiple layers of encoder that takes the input from the previous layer of encoder. Decoders take the input from the previous layer of decoder, and also the output of the last layer of the encoder. 
- Decoder generates words as we go. 
- **Encoder**: self-attention -> feed forward neural network 
  - (independent feedforward nets for each head)
- **Decoder**: self-attention -> encoder-decoder attention -> feed forward
  - Encoder-decoder attention: get the K and V from the encoder. 
- Last layers:
  1. Linear layer
  2. Softmax: get the probabilities of which word should be in this position

## Alternatives to Attention

**State-Space Model**

- $y_k = CA^kBu_0 + CA^{k-1}Bu_{k-1} + ... + u_k$

  - [u0, u1, ..., uk]: input

  - y: output

- It's a convolution, but the calculation is still the same complexity (quadratic). 

- Signals and systems class -- Convolution in the time-domain is element-wise multiplication in the frequency domain. 

  - i.e. need to change from time domain to frequency domain to reduce the complexity. 
  - use FFT. $O(L\log{L})$
  - With this complexity, we can afford long imput that transformer can't afford to calculate. 


**Using SSMs as Layers**

- Directly training this will make it have the same issue with RNN. Need to do special things.

 *S4 (Structured State Space Models)*

- Stack multiple SSMs together
- Good: can handle long sequences
- Bad: can't handle "selective" tasks (e.g.selective copying, not copying a fixed sequence, but the position is also given in the input. like llm summarizing a paper. )

Manba model: selective SSM

- Let B and C use input.
- Breaks the convolution, so breaks the FFT.
- Can use scan approach to do a relatively fast recurrence calculation. 
- Can handle selective copying as well. 





## Training Transformers

- ==Cross entropy==
- "Self-supervised -- the original transformer is not. It's just supervised and we're lucky that we have a ton of data. 

## Encoder/Decoder Only

**Encoder-only**: good for classifiers

- Contexual embeddings: take the context into account to produce embeddings for word (e.g. "bank" has several meanings)
- **BERT**: bidirectional: we can look the entire inpu/context, not just previous input. both before and after the word. 
  - Stack encoders. Pad input with a special [CLS] token, so when going through a bunch of encoders, this CLS can be output as some classification vectors (e.g. sentiment analysis
  - So doesn't need to take all the embeddings and put into another layer to produce classification
  - **Training**: Pretraining on fake tasks, then fine tune on real tasks
    - **Pretraining tasks**: Need to pretrain on both tasks, so we get the capability we want. 
      - **Masked Language Modeling**: guess what's at the masked word. Forcing the model to focus on context. 
        - Add noise. Words at random. Sometimes no mask at all and flip word randomly. 
      - **Next Sentence Prediction**: if the next sentence fits or not. 
  - **Finetuning**: actually train on the task.
    - SQuAD: question and answering. 
    - 

#### **Decoder-Only**

- Only decoders
- "Auto-regression": plug output back into the input

##### Llama 3.1

- SwiGLU
- RoPE
- Attention mask to prevent attention between different documents in a sequence
  - Sometimes one document will not fill out the entire context window, so put some other documents after this. 
- **Groped query attention**: instead of values, keys, queries for each token, have one key and one value for multiple queries.
  - Multi-head: one-to-one
  - Grouped-query: one-to-many
  - Multi-query: one-to-all
- Vocabulary with 128K tokens: a standard tokenizer + 28K additional tokens for non-English 

##### Hybrid Models: Attention + SSM

- Transformer + Mamba
- High throughput 
- Benchmarks: Accuracy (answering multiple choice questions)
- Throughput: Output tokens/s/GPU
- Weird: base model (56-B) -> pruning to get a smaller model (47-B) and gets better accuracy and performance



# Prompting

- **Zero-shot**: no "examples" provided to the model
  - **Hard-prompting**
    - Finetuning on one task can be transferred to relevant tasks
    - Choose/optimize the prompt. Search for one single prompt that works the best
  - Soft-prompting (continuous prompting)
    - Insert non-language parameters into prompt
    - Tune these token prefixes to specific task. E.g. prefix for table-to-text, prefix for summarization, etc
    - Only store these specific parameters. 
    - Have to actually train these tokens
  - Ensembling prompts:
    - Get multiple samples of the prompt. (e.g. change temperature, vary prompt). Combine output. (majority vote - "Self-consistency")
    - Chain-of-thought: 
- **Few-shot/in-context learning**: provide some examples
  - "few shot" can sometimes also mean finetuning, but sometimes also means prompting (no learning)
  - in-context learning is not learning. Always means examples in promping
  - Structure effects:
    - Format 
      - Very old: masked language model: filling in blank
      - Left-to-right: prefix prompt
        - An old Eval datasets have pre-created prompts
      - Recent: instruction tuning
        - Fine tune 
        - More natural way to tell the model. Challenging to search for ways of instructions.
    - Choice
    - Order

### **Discrete Optimization **(Hard prompting)

- Greedy: start from a prompt, break it up into phrases, use standard nlp tools, operations (delete/swap/add) to get candidates, evaluation the score and choose the best candidate, and then go back. 

- **Auto-Prompting**
  - Use another LLM to create candidate prompts
  - One idea: **LLMs as "optimizers"** -- guess the solution of a optimization problem. Evaluate them externally. Run in a loop with few-shot.
    - Meta instructions, examples + scores, problem to be solved
  - Basically all methods that can evaluate the neighbors: hill climbing, simulated annealing, genetic algorithms, etc
    - Promptbreeder: a way of genetic algorthms. 

### **Prompting VLMs and Multimodal Models**

- **CLIP-style VLMs**: a text encoder + an image encoder. Show you embeddings and tell you if they match or don't match. 
  - Vector: parts of image. Vector: parts of text. 
  - If they match, the dot product is high; If they don't match, the dot product is low. 
  - We want to let the diagonal highest. So each part of text represents each part of image. 
  - **Pretraining**: Text encoder & Image incoder. We have descriptions and images in the dataset. 
  - **Create classifier**: add a mask. and then see what embedding option in all the candidate labels has the biggest dot product at the masked position with the image encoding. 
  - Standard way: use pre-defined prompts lile "a photo of [X]"
  - We want more information in this. So don't use completely static descriptions. Use more descriptive features. -- tiger -> stripes, claws, ...
  - How to annotate all the features? Do it with an LLM. 
  - **Spurious Features**: Sometimes two things usually go together in pictures, so LLM will mistaken one as another. e.g. waterbirds vs. landbirds. So model learned that look at the background. 
    - Btw the LLM can tell you what's going on here. So you can ask the LLM to ask what's going on. 
      - "hi llm, tell me what's some possible spurious features in this setting."
    - We can modify the embeddings to get rid of these spurious features. 
  - **In-Context learning for VLMs**
    - e.g., ask a VLM (not CLIP. This VLM needs to be generative) to do outline/annotation of input image. The example in few-shot learning you give the VLM. 

### **Chain-of-Thought**

- Usually LLMs don't have "scratch paper". So we want them to have this room for thoughts/intermediate steps.
- The basis of reasoning-based models today
- **Zero-Shot chain-of-thought**: no example, "let's think step by step"
  - Prompt for answer extraction. 
- **Few-Shot chain-of-thought**: provide some examples of how to think step-by-step. More structured. 
  - We can also run MV (majority vote) over the outputs.
  - Even if you show the model a wrong logic in CoT example, it doesn't really change the performance of the model. The point is to show the structure of thoughts. Like steps. 
- **Generalizations**
  - "**Tree of thoughts**": We reason by going down a tree of options, and pick one of the chain as the final answer. 
    - Generate a thought of tree: generate candidates for next step
    - Run tree search 
- Small models don't have CoT behavior no matter how you prompt it. It shows up only at a certain size. 
- Reasoning data in pretraining doesn't entirely extend to other languages. 
- CoT doesn't always make things better. It helps math and symbolic, but make text classification, context-aware QA, multi-hop QA etc worse. So not always want to do. 
- Today it's not usually something you can control if you want to use CoT or not. The model will decide. but some open source models still allow you to control. 

# Fine-Tuning and Adapter

## Tools

### Program-aided LMs

- Language Models struggle to do arithmetics because it do math problems using tokenizers and try to predict, like in a language way. 
- So we should give it a calculator and let it know how to call it and use the answer. 
- Or: let it write a line of python code for each step, and use a python interpreter. 
  - Modern language model will figure out when to use it. 
- **Program-of-thoughts**: similar idea. 

### General Tools

**Toolformer**: API calls to models

- Old technique: fine tune on API call examples
- Model context protocol standardize tool calls

## Fine-Tuning

**Fine-Tuning vs. Prompting**: sometimes in-context learning can be worse computational wise than fine-tuning. 

- In-context learning scale worse, more FLOPs but less accuracy. 

**Frozen models**

- fine tune only some part of the model on top of our foundation label. 
- e.g. linear probing. i.e. single linear layer. 

**Full Fine-Tuning**

- If the task is too different from linear probing, we can do a full training from the base model. 
- As expensive as training a full model.
- Issues on OOD (Out of distribution) data: Easy overfit. 
  - Problem: Full Fine-tuning bad at OOD. Linear probling is slightly worse at ID test but better at OOD. 
  - **LP-FT**: LP first, then FT. This makes both ID and OOD good. 

**Partial Fine-Tuning**

- Literally that
- Doesn't need to be the top layers. (Only top perfomance isn't great)

## Adapters

**Parameter-Efficient Fine-tuning (PEFT)**

- Example: **Prefix-Tuning**

  - For soft-prompting.

  - Not touching the weights, but instead change the prefix token we put in fromt of the prompt for each task. This way we can switch the token when we're switching tasks -> Enable cheap adaptation. 

  >  ==Prefix-Tuning vs Prompting. Why do we still want prompting techniques, especially zero shot?==
  >
  > Even when we are doing prefix tuning or frozen layers, we still need to do back propagation through the weights to calculate the gradient, so we need access to the weights of the model when we're doing prefix tuning. However, we can do prompt engineering even when we don't have access to the weights. 

  - An example of 

- Training some **Adapters** can train few weights and perform better than top layers.

  - Adapters are small, so we can switch them for different tasks.
  - Inserted this new module in between model (e.g. multi-head attention -> ffl -> adapter)
  - It's not new ('19')

- **LoRA**: Low-Rank Adapters

  - Most popular, cheap
  - Not adding a separate module anymore. Instead, add something to the pretrained weight. For performance to be better, Same dimention but fewer prarmeters -> low-rank matrix. Multiply these matrices to get the same dimention matrix to add to the pretrained weights matrix. 
  - Rank: how many non-zero. The weights might be close to low rank -> some value is really low, essentially zero. Like sparse matrix? 
  - Manifold 
  - These LoRa were initially only added to the attention layer, but now we can put it anywhere. 

## Cross-Modal Adaptation

- Intuitively, there are something abstract going on that can be adapted to any modal. 
- Basic idea: change input embedding and output layer. 

### ORCA

- Source dataset: the modal that the model's pretrained on (x^s, y^s)
- Target dataset: the new dataset to be adapted to (x^t, y^t)

1. Dimensionality alignment: 

   - Task-specific embedder -> Body -> task-specific predictor
   - make the target data compatible with the pretrained model (in terms of dimension)
   - Input: convolution for image.
   - Output: pooling and linear classifier. 

2. Distribution Alignment

   - learn the task-specific embedder to make the distribution align with the input that the model already kinda knows - looks more similar to the source dataset. 

   - > Model might be more familiar with some part of the space, not all over the dimension. 
     >
     > So let's make sure we're sending the target data to somewhere very similar. 
     >
     > ==why are we considering y when we're changing the distribution of x?== 

   - Define a "distance" between source dataset and target dataset: Learn the embedder f^t to minimize the distance between $(f^t(x^t), y^t)$ and $(f^s(x^s), y^s)$

     - It's not difficult to define distance between embeddings, but the outputs can be very different (for different tasks)
     - So need a distance function: **Optimal Transport**
       - We find a "move" function that transform the distribution
       - And we find the "move" function that has the min cost
       - Instead of talking about y ("labels") directly, we replace y with P(X|y) (distribution of vectors that lead to the predictions)
       - We use **Wasserstein**: $W(P(X|y), P(X|y'))$
     - We break down the OTDD into 1. distribution of inputs and 2. the Wasserstein we just defined. 

     > ==Still confused about how exactly Wasserstein is computed.==
     >
     > ==class conditional distribution==

3. Fine tune the input and output network weights

- Result: Changing distribution alignment works pretty well: it can even beat traditional Neural Architecture Search (NAS) i.e. find the best architecture. 

# Alignment

"Do the users like it? "

Use reinforcement learning. But how? 

**Reinforcement Learning from Human Feedback**: RLHF

Don't change the model directly, but train another reward model based on (prompt, winning_response, losing_response). This reward model is also a language model, that for a particular generation, it can output score. 

1. Get feedback

   - Produce different responses for the same prompt (the responses can be produced from the different models or the same model)
     - Problem with different model: too different, don't do much

     - Problem with the same model: hard to do the RL training in real time

   - Ask human which is better (binary output), or ask them to give a score (hard to aggregate because different raters have different styles, like 2% 5's or 30% 5's)

2. Preference mode

   - train this reward model to give scores for different responses

   - Preference model: Bradley-Terry model

   - Loss: based on the log likelihood

     > ==log likelihood==

   - You can start from pretrained reward models. There's a benchmark: rewardbench

   - This model has to be another language model, but produce a single scale value

3. Fine tuning with RL

   - Action space: all the tokens we can output
   - State space: sequence tokens we have seen
   - Reward function: the trained reward model
   - Policy: the new version of the language model we're training. Hopefully it will output a higher reward generation. 
   - **Proximal Policy Optimization**: try to make it get better rewards, but at the same time, define a "penalty" of changling too much from the original model, so it won't overfit on rewards. 
     - We calculate the D_KL, which is comparing the possibility distribution of the next token generated. 

- Problem with too much RLHF: model starts to refuse responding to anything

Why RL: (Hypothesis)

- For knowledge-seeking interaction: SL failed to stop model lying when it doesn't know: it's not in the dataaset. Also, It's hard to put "i don't know" in the training set because it will only encourage them to reply "i don't know" to that specific question
- RL encourages truth telling. Most of the time if it makes things up, it will lose reward. 
- **Abstains**: craft a reward model to encourage abstains
  - High reward: correct
  - Medium reward: abstain
  - Negative reward: incorrect

RLHF Problems: 

1. Human Feedback
   1. Bias, data poisoners, lack of care
   2. What is "good" output? 
   3. High quality data is hard to get: good prompts that the users would actually ask
2. Reward model
   1. Friendlines, helpfulness etc. Depends on context, different types, can't do universally to all users, but personalizing RLHF is really expensive
   2. Reward hacking. e.g. May prefer longer response because in training set, longer is preffered because the shorter one is dumb, but that's not necessarily always the case



Alternatives

​	DPO	RLAIF



RL outside of alignments

- Not to make human happy, but to solve some problems
- Reward? There's not a single correct answer. Doesn't have to match every single tokens. 
- Verifiers:
  -  e.g.  check the final answer for math problems; write unit tests to test generated code; other reasoning
  - We have a lot of data in math and code. 
  - 

PPO

GRPO

### Efficient Training

Flash Attention: Use faster memories to optimize training time.  

- Tiling

### Efficient Inference

Autoregressive models, you need to wait until the last token is generated to start generate the next one

- Why not use not autoregressive models (e.g. diffusion models)? So it can generate the entire sequence at the same time. 
- Another idea: **Speculativve Decoding** offload generation to a faster model, and use the original model only to check the generation, which can be done in parallel, and the speed is equivalent to using the original model to generate one token. 
- **Adaptive Language Modeling**: sometimes don't wait until the end of all layers. Cut off layers and generate the token when it's safe. 
- **Parallelizing Decoding: Medusa Decoding**: generate top-k candidates for each position, and then assemble into a full sequence 







### Agents

Building blocks:

input

Memories

1. Long context windows
   - It used to be not enough
   - Context window can be structured
   - Long context windows: It will lose information. The model pays a lot of attention to the beginning and the end, and sometimes middle is lost

Reasoning and Planning

- Including older methods (e.g. CoT) e.g. ReAct: thought, act, observation

Tool

- Tool Noise/Uncertainty/Tool use errors
- Creating New Tools: A powerful LLM create new tool, and then use a lightweight model to run the agent in long term





