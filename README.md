## _LLM Agents for Tool Planning and Function Calling - Part 1_

This post is based on my work building efficient and reliable LLM agents for external tool usage. I will start with a brief intro on reasoning and planning with LLMs, and then we will move on to other cool stuff like tool usage and function calling with LLMs. The work started as one of the problem statements proposed by DevRev at Inter IIT Tech Meet 12.0, and later, we extended our work into a research paper. Here is the arxiv preprint of our work: **[SwissNYF: Tool grounded LLM Agents for Black Box Setting](https://arxiv.org/abs/2402.10051)**

Ps: I will use tool/function/API interchangeably throughout this post.

### **Reasoning & Planning with LLMs**

Planning involves breaking down a complex problem into simple and sequential subproblems and solving them individually. These subproblems can be independent or sequential. In the latter case, the output of the previous steps is required as input to a current or subsequent step. Solving these subtasks involves some kind of reasoning, such as mathematical reasoning, performing operations, algebraic manipulation, etc., or conceptual or factual rationale. These reasoning are generally elicited from the internal memory of the LLMs based on the vast amount of data they are pre-trained on and also the different downstream tasks they are finetuned on. For example, you can finetune an LLM on various math problems to improve its mathematical reasoning abilities.

​                    **Mathematical Query:**

<img src = "https://github.com/user-attachments/assets/42c8720e-0d70-4000-a43c-39cb672bc5fe" width = 800/>

​                     **LLM Reasoning:**

<img src = "https://github.com/user-attachments/assets/a2216210-ca21-472e-bfa0-92fc3f8bbaf6" />


However, eliciting this reasoning is challenging as sometimes these LLMs tend to hallucinate. What is hallucination? Sometimes, LLMs perform the wrong reasoning or use a flawed concept in their solution, resulting in an incorrect answer. Still, they do it with confidence, which means they are not sure of the accurate reasoning, resulting in a false one, leading to a hallucinated answer. 

Different techniques have been developed to help improve LLM reasoning. Sometimes, one can finetune an LLM for a specific reasoning task. However, this requires a good amount of computing, which is not readily available. Other methodologies propose different prompting techniques. Detailed instruction helps LLM think better and provide more accurate reasoning. There has been a lot of work going around in this space. We will discuss some of them in the context of Tool Planning.

### **Tool Planning & Function Calling**

What is Tool Planning? Let's imagine you are going through the documentation of a framework. Let's take *PyTorch* as an example. Let's say you need to generate two *1D* tensors of *dim = 3*, one with all *0s* and the other with all *1s*, and then you have to perform the dot product of these two. You are still very naive with *PyTorch* and want to use a torch chatbot to help you. You give your query to the chatbot, which has access to the complete *PyTorch* documentation. This is a task of tool planning & function calling.

First, the query will be passed to a retriever, which, using some kind of similarity metric, will narrow down the documentation into, let's say, *2-3* pages containing relevant torch functions that may help solve the query. The role of the retriever is to narrow down the search space for the LLM while performing planning and reasoning. This is generally based on similarity between the embeddings of the documents and the user query.

<img src = "https://github.com/user-attachments/assets/93ce920d-6308-470e-bc4e-ad9e2c78e7a9" width = 800/>


Now the query and these retrieved pages are given to an LLM (Planner) with some customized prompt, and the LLM has to select the relevant functions from the retrieved pages, decide the order in which they must be implemented, and choose the appropriate argument values for each of these functions to address the query completely. Deciding this order of implementation is what we refer to as Tool Planning, and filling the appropriate argument values is what we call function calling. 

Here is the step-by-step breakdown of the tool planning and function calling for above torch example:

1. **Tool Planning**

   - Identify the functions needed: `torch.zeros`, `torch.ones`, and `torch.dot`.
   - Plan the sequence of function calls.

2. **Function Calling**

   - Call `torch.zeros` with argument `3` to create a tensor of zeros.

   - Call `torch.ones` with argument `3` to create a tensor of ones.

   - Call `torch.dot` with the two tensors created above as arguments.

     

### LLM Reasoning Methodologies

There have been a lot of work that has been done and continues to be in improving the reasoning abilities of LLMs for various different tasks. Building reliable and efficient LLM Agents can help in various direction: Automating software development, Educational Purpose etc. I'll talk about some of the common approaches used.

#### Chain of Thought Reasoning

Introduced in [Wei et al. (2022)](https://arxiv.org/abs/2201.1190), chain-of-thought (CoT) prompting enables complex reasoning capabilities through intermediate reasoning steps. The idea is to instruct the LLM to breakdown a complex problem into subproblems which individually are much asier to solve and use these solved sub answers to address the original question. It has been one of the most used reasoning techniques and almost all the papers proposing other alternatives use CoT as the baseline.

<img src = "https://github.com/user-attachments/assets/33f12494-5bc7-46cc-9f9c-82080a48f2c1" width = 800/>


#### ReAct

[Yao et al., 2022](https://arxiv.org/abs/2210.03629) introduced a framework named ReAct where LLMs are used to generate both *reasoning traces* and *task-specific actions* in an interleaved manner. React uses the *Thought*, *Action*, & *Observation* patterns in each step to perform coherent reasoning paths, eventually leading to the required answer. First, the LLM generates a thought for intermediate steps; for each thought, it selects one of the possible actions and gets an observation. For example, in the case of tool planning, LLM will think of using a particular tool (*thought*), which would then be called upon with appropriate arguments (*Action*), and then the tool response will be observed (*Observation*). This way, LLM can determine whether a particular tool is relevant to the user query. Here's an simple example of a React agent solving a simple maths problem using functions such as *multiply* & *add*.

<img src = "https://github.com/user-attachments/assets/6ca472af-d7b3-407a-b4d1-278e537bccae" width = 700/>


#### **DFSDT**

Depth First Search-based Decision Tree (DFSDT) is a method used for solution path annotation that can handle multiple tools, calls, responses, and their errors. It was introduced by [Qin et al., 2023](https://arxiv.org/abs/2307.16789). DFSDT serves as a general decision-making strategy to enhance the reasoning capabilities of LLMs. It works by running a Depth First Search over a Tree of Thought which has several branches of *Thought* & *Action*. It keeps a track of errors encountered and broadens the search space by taking new actions while considering errors in previously executed chains. This ensures that multiple reasoning traces are explored, and the search is not jeopardized by any single failure. By expanding the search space, DFSDT can better solve those difficult, complex instructions that are unanswerable by the vanilla ReAct no matter how many times it is performed. At the same time, in hindsight, it may take an extremely long time to figure out the right chain of thought and end up spending a large number of tokens on failed trajectories.

![WhatsApp Image 2024-07-13 at 02 35 25](https://github.com/user-attachments/assets/ef40ec02-343b-4e0b-891f-abba7da14e34)


These are some of the methodologies explored for enhancing the reasoning & planning abilities of LLMs in the context of Tool Reasoning. New approaches are being investigated, and much research has been done in this space. All the above three methods are traversal-based, i.e., they start with a root node and perform a search on the next relevant step, finally leading to a final answer. CoT and ReAct perform the next step search with a sample size of *1*, whereas in the case of DFSDT, the sample size is *>1* for the root node and *1* for all the subsequent nodes. Here, each node represents an intermediate step.

This is all for part 1; I hope this gives you a brief idea about reasoning within LLMs and what it means in the context of Tool LLM Agents. In the next part, I'll go through the curious case of Black Box Setting with these tool agents and discuss our approach to tackling this exciting and vital scenario as presented in our work.
