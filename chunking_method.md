Chunking Strategy

Prototype: Naive fixed-size splits (e.g. 512 tokens)
Production: Semantic chunking, hierarchical chunks (parent-child), document-aware splitting that respects structure (headers, tables, code blocks)

Text chunking is one of the most critical steps in building a RAG (Retrieval Augmented Generation) pipeline. How you break up your documents directly impacts the quality of your entire system. A poor chunking strategy can lead to irrelevant context being inserted into your prompts, causing your AI to give completely wrong answers.Let's explore three main approaches.
1. Size-Based Chunking
Size-based chunking is the simplest approach - you divide your text into strings of equal length. If you have a 325-character document, you might split it into three chunks of roughly 108 characters each.This method is easy to implement and works with any type of document, but it has clear downsides: Words get cut off mid-sentence, Chunks lose important context from surrounding text, Section headers might be separated from their content.To address these issues, you can add overlap between chunks. This means each chunk includes some characters from the neighboring chunks, providing better context and ensuring complete words and sentences.
2. RecursiveCharacterTextSplitter: tries to cut at natural boundaries
It takes a list of separators, ordered from "best break" to "worst break":


separators = ["\n\n", "\n", ". ", " ", ""]
              ↑        ↑      ↑     ↑    ↑
          paragraph  line  sentence word char
3.  Structure-based chunking divides text based on the document's natural structure - headers, paragraphs, and sections. This works great when you have well-formatted documents like Markdown files.This approach gives you the cleanest, most meaningful chunks because each one represents a complete section. However, it only works when you have guarantees about your document structure. Many real-world documents are plain text or PDFs without clear structural markers.
4. Semantic-based chunking is the most sophisticated approach. You divide text into sentences, then use natural language processing to determine how related consecutive sentences are. You build chunks from groups of related sentences. This method is computationally expensive but produces the most relevant chunks. It requires understanding the meaning of individual sentences and is more complex to implement than the other strategies.
5. Sentence-Based Chunking: A practical middle ground is chunking by sentences. You split the text into individual sentences using regular expressions, then group them into chunks with optional overlap.



Your choice depends entirely on your use case and document guarantees:
Structure-based: Best results when you control document formatting (like internal company reports)
Sentence-based: Good middle ground for most text documents
Size-based: Most reliable fallback that works with any content type, including code
Size-based chunking with overlap is often the go-to choice in production because it's simple, reliable, and works with any document type. While it may not give perfect results, it consistently produces reasonable chunks that won't break your pipeline.


Semantic Search: The most common approach for finding relevant chunks is semantic search. Unlike keyword-based search that looks for exact word matches, semantic search uses text embeddings to understand the meaning and context of both the user's question and each text chunk.

Text Embeddings: A text embedding is a numerical representation of the meaning contained in some text. Think of it as converting words and sentences into a format that computers can work with mathematically.

You feed text into an embedding model
The model outputs a long list of numbers (the embedding), Each number ranges from -1 to +1. These numbers represent different qualities or features of the input text.Each number in an embedding is essentially a "score" for some quality of the input text. However, here's the important caveat: we don't know precisely what each number represents.The actual meaning of each dimension is learned by the model during training and isn't directly interpretable by humans.


Hybrid Search Strategy
The solution is to run both semantic and lexical searches in parallel, then merge the results.
Semantic search finds conceptually related content using embeddings
Lexical search finds exact term matches using classic text search
Merged results combine both approaches for better accuracy

BM25 excels at finding exact matches because it:
Gives higher weight to rare, specific terms,Ignores common words that don't add search value,Focuses on term frequency rather than semantic meaning.Works especially well for technical terms, IDs, and specific phrases.
The key insight is that both search methods have complementary strengths. Semantic search understands context and meaning, while lexical search ensures you don't miss exact term matches. By combining them, you create a more robust search system that handles both conceptual queries and specific lookups effectively.