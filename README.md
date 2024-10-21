# LIKR：LLM's Intuition-aware Knowledge graph Reasoning model
Here is the official implementation of our LIKR model, which is an LLM+KG-based recommendation model. 
Currently, this repository is anonymous, but we plan to lift the anonymity after the associated paper is accepted.

### Dataset creation
Implement the folders in the `dataset_making` directory in the following order:
   ```bash
   python 1_gpt_api_ml1m.py
   python 2_make_eid_output.py
   python 3_make_newid_output.py
   ```
Please place the final generated `newid_output` folder inside the `model` folder.

### KG Reasoning
Implement the folders in the `model` directory in the following order:
   ```bash
   python train_transe_model.py
   python train_agent.py
   python test_agent.py
   ```

## Acknowledgment
This code is based on the implementation described in the following paper. We extend our deep gratitude to the authors of the prior work.
```
Balloccu, G., Boratto, L., Cancedda, C., Fenu, G., Marras, M. (2023).
Knowledge is Power, Understanding is Impact: Utility and Beyond Goals, Explanation Quality,
and Fairness in Path Reasoning Recommendation. In: , et al. Advances in Information Retrieval.
ECIR 2023. Lecture Notes in Computer Science, vol 13982. Springer,
Cham. https://doi.org/10.1007/978-3-031-28241-6_1
```
