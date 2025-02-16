import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { Document, DocumentInterface } from "@langchain/core/documents";
import { PromptTemplate } from "@langchain/core/prompts";

export const loadAndSplitTheDocs = async (filePath: string) => {
  // load the uploaded file data
  const loader = new CSVLoader(filePath);
  const docs = await loader.load();

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 0,
  });
  const allSplits = await textSplitter.splitDocuments(docs);
  return allSplits;
};

export const vectorSaveAndSearch = async (
  splits: Document[],
  question: string
) => {
  const embeddings = new OllamaEmbeddings({
    baseUrl: "http://localhost:11434", // Default value
    model: "deepseek-r1:1.5b",
  });
  const vectorStore = await MemoryVectorStore.fromDocuments(splits, embeddings);

  const searches = await vectorStore.similaritySearch(question);
  return searches;
};

export const generatePrompt = async (
  searches: DocumentInterface[],
  question: string
) => {
  let context = "";
  searches.forEach((search) => {
    context = context + "\n\n" + search.pageContent;
  });

  const prompt = PromptTemplate.fromTemplate(`
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
`);

  const formattedPrompt = await prompt.format({
    context: context,
    question: question,
  });
  return formattedPrompt;
};

export const generateOutput = async (prompt: string) => {
  const ollamaLlm = new ChatOllama({
    baseUrl: "http://localhost:11434", // Default value
    model: "deepseek-r1:1.5b", // Default value
  });

  const response = await ollamaLlm.invoke(prompt);
  return response;
};
