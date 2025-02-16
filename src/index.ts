import Express from "express";
import {
  generateOutput,
  generatePrompt,
  loadAndSplitTheDocs,
  vectorSaveAndSearch,
} from "./rag";
import path from "path";
import { OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

const run = async () => {
  const app = Express();

  // Using a test dataset for fast tests
  const splits = await loadAndSplitTheDocs(
    path.join(__dirname, "../datasets/Leagues/EnglishPremierLeague.csv")
  );

  const embeddings = new OllamaEmbeddings({
    baseUrl: "http://localhost:11434", // Default value
    model: "llama3.2",
  });

  app.use(Express.json());

  app.get("/health", (req, res): any => {
    return res.status(200).json({
      message: "I'm healthy!",
    });
  });

  app.post("/prompt", async (req, res): Promise<any> => {
    const question = req.body.question;

    const searches = await vectorSaveAndSearch(splits, question);

    const prompt = await generatePrompt(searches, question);
    console.log(prompt);
    const result = await generateOutput(prompt);
    return res.status(200).send({
      content: result.content,
    });
  });

  console.log("Running in http://0.0.0.0:8080");
  app.listen(8080);
};

run();
