import 'dotenv/config';

import readline from 'readline';

import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { HuggingFaceTransformersEmbeddings } from '@langchain/community/embeddings/hf_transformers';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const prompt: () => Promise<string> = async () => {
  return new Promise((resolve) => {
    rl.question(
      'ChatPDF: What would you like to know?\nYour question: ',
      (answer) => {
        resolve(answer);
      }
    );
  });
};

const start = async () => {
  // Change the path to the PDF file you want to use
  const pdfPath = './pdf/context.pdf';
  const pdfLoader = new PDFLoader(pdfPath);
  const textSplitter = new RecursiveCharacterTextSplitter();

  const documents = await pdfLoader.load();

  const pages = await textSplitter.splitDocuments(documents);

  const embeddingModel = new HuggingFaceTransformersEmbeddings({
    modelName: 'Xenova/all-MiniLM-L6-v2',
    onFailedAttempt: (error) => {
      console.error('Failed to generate embeddings', error);
    },
  });

  const db = await FaissStore.fromDocuments(pages, embeddingModel);

  const llmProvider = new ChatGoogleGenerativeAI({
    model: 'gemini-pro',
    maxOutputTokens: 2048,
  });

  let query = '';

  while (true) {
    query = await prompt();

    if (query === 'quit') {
      console.log('Goodbye!');
      process.exit(0);
    }

    const relevantDocuments = await db.similaritySearch(query);

    const context = relevantDocuments.map((doc) => doc.pageContent).join('\n');
    const enhancedPrompt = `Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.----------------\nContext:${context}\nUser question:\n${query}`;

    const response = await llmProvider.invoke(enhancedPrompt);

    console.log('\nChatPDF: ', response.content);
  }
};

start();
