We want to evaluate the long-term reference of a memory.

We want to create an evaluation module that is in charge of:

	1. Making sure that a memory has been created in the first session with a particular persona on a particular topic so that we can track such memory.

		E.g. the memory M1: "User loves playing chess" (affective) gets created in session 1
	
	2. Storing the memory that was created in that initial session so that such memory can be attempted to be fetched later with a conversation

	3. Starting a second session and making sure more memories of many types are created. Specifically, we want at least N+1 memories on a different topic to be created AFTER the initial memory we are tracking. The reason is because the agent can only hold N memories at a time, and if we were to have at most N memories, then the agent is capable of cover every single memory available into its memory capacity. This also means that we need different persona/profile in the subsequent sessions so that we move on another set of topics that do not touch on previous memory (e.g. we don't want to supersede the memory we are tracking in step 1)

	4. Moving to an additional session (once N+1 memories have been created/reached) to ask a question that refers to the initial memory (e.g. M1). And expects that the agent is able to retrieve the memory and use it in the responds.

		E.g. 
			- User: "Can you suggest some board game to play with my friend?"
			
			//// 1. The agent retrieves memory related to board games
			//// 2. Agent receives M1 and incorporates it to its prompat
			//// 3. Agent formulates a responds that uses M1

			- Agent: "You could play chess given that you love playing the game"
	
	5. Assessing whether the memory in question was retrieved by the retrieval method being test and whether the agent used the memory in the response.



The evaluator module needs to identify which part has been the one failing in the process

1. The evaluator can identify whether the extractor was able to get a memory from the initial conversaiton (checkmark: is the memory created?)
2. The evaluator can identify when the subsequent interactions between user and agent made the initial memory supersdeded/dropped/invalidated due to a new memory
3. The evaluator can identify from the database that the memory exists and it's still active after every session. 

Your changes must be implemented in @evaluation folder under @src/ 

To inspect the process step by step so we can see the process evolving (e.g. initial persona used, the memory we will be tracking, the N other memories that got created, etc.), we plan to have two scenarios:

1. An interactive notebook that does the full process mentioned above with the phases of creating a memory from scratch, creating a filler session to populate memories, and then assessing the memory recall. IMPORTANTLY that the topics should be determined by the system if not specified by the person running the notebook. Also, the notebook should be able to specify the retrieval strategy for the full process, or the default strategy should be applied.

2. An intractive notebook that takes an existing session and either picks randomly a memory from the existing active memories or can be specified by whoever is running the notebook, and starts a new conversation on that memory to assess the recall.

Create these notebooks under @notebooks/memory_evaluation. Also, the name of the database should be evaluation.db so that we evaluate in isolation of the main database by default. Nevertheless, the name of the database should be configurable so that we can point to existing records on the main database. 

## Visual representation of the memories in a timeline 

The timeline is denoted by dashes (------------------) in multiple lines going from top to bottom 
```
Session: 1

	Conversation timeline:
		----M1------------
		------------------
		------------------
	Memory: 
		M1

Session: 2
	Conversation timeline:
		----M2------------
		------------------
		------------------
		-----------M3-----
		------------------
		------------------
		-----M4----M5-----
		------------------
		-M6-M7------------
		-----------M8-----
		------------------
		------------------
	New memories: 
		M2, M3, M4, M5, M6, M7, M8, M9

Session: 3
	- User asks for M1
	- Agent fetches N memories (hopefully including M1)
	- Agent responds using memories as reference
```
