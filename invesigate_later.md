# Questions can be explored later (doubts during building)

1. Do I really need breadcrumbs to preserve the section for better retrieval? Or can I just use recursive character text splitter directly because it has metadata that include section(is there a index or something?)

2. Is it a problem that each section doesnot have any overlapping after chunk by header? There is overlap between the chunks within each section but not between each section.

3. When decide which libraries to use, how do I make sure they are compatible?

4. How to record all the major metrics like how many chunks with different chunk strategies,queries similarity search for chunk rank before BM25,Hybrid

5.chroma_db: current workflow is if chroma_db folder exist, then add the embeddings, if not then create the folder.

6. HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base") can I select other models? How to decide which models to select?

7. About the pdf file load - what does it mean and what is the limitation?

8. Helper function or libraries, which one? When to use?

9. Local llm choices

10. converted pdf to markdown files - missing the page metadata, what does that mean to the retrieval?

11. Engineering judgment includes knowing when to stop. How do I know if my eval test with retrieval tuning is enough - the metrics has improved? 

-1. Recall@15 = 1.00 — is the answer findable at all within the top k window?
-2. Overall MRR: 0.72 → 0.76 with hybrid search, how high up did the answer appear?
-3. MRR per query type:
• Indirect: 1.00
• Definition: 0.78
• Enumeration: 0.49 → 0.72 with hybrid (findable but buried, lists don't chunk well by default)
• Negation: 0.33 → 0.50 with hybrid (same pattern, list items split across chunks)
• Negative (no info type): excluded from MRR, no correct chunk to rank


12. What happen everytime when I restart the kernel, what does that mean to every function I created?

13. What is vectorindex? what does it do to my current vectorstore?

14. What does it mean to change the embedding model to a bigger one mean?

