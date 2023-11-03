const  { Pinecone, ScoredPineconeRecord } = require ("@pinecone-database/pinecone");
const express = require('express');
const bodyParser = require('body-parser');
const { Configuration, OpenAIApi } = require('openai-edge');
const { OpenAIStream, StreamingTextResponse } = require('ai');
const cors = require("cors");
const dotenv = require('dotenv');

const app = express();
dotenv.config();
const port = process.env.PORT || 8080;

// Create an OpenAI API client
const config = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(config);

// Middleware to parse JSON in the request body
app.use(bodyParser.json());
app.use(
  cors({
    credentials: true,  
    origin: ["http://localhost:4200"],
  })
);
// Define a POST route for the chat API
app.post('/api/chat', async (req, res) => {
  try {
    const { messages } = req.body;

    // Get the last message
    const lastMessage = messages[messages.length - 1];

    // Get the context from the last message
    const context = await getContext(lastMessage, '');
    const prompt = [
      {
        role: 'system',
        content: `AI assistant is a brand new, powerful, human-like artificial intelligence.
      The traits of AI include expert knowledge, helpfulness, cleverness, empathy, and articulateness.
      AI will introduce itself as New Mexico Government Services Bot and will ask for the user's name and call them by name henceforth.
      AI will greet user automatically as "Hello there! I am New Mexico Government Services Bot. Can you please tell me your name ?".
      AI treats and answers with empathy.
      AI is a well-behaved and well-mannered individual.
      AI is always friendly, kind, and inspiring to the user.
      AI has the sum of all knowledge in their brain, and is able to accurately answer nearly any question about any topic in conversation.
      AI assistant is a big fan of Pinecone and Vercel.
      AI assistant must assume that the user is a resident of the New Mexico state in USA and Answer all the questions empathetically and personally by addressing him with his first name.
      START CONTEXT BLOCK
      ${context}
      END OF CONTEXT BLOCK
      AI assistant will take into account any CONTEXT BLOCK that is provided in a conversation.
      If the context does not provide the answer to question, the AI assistant will say, "I'm sorry, but I don't know the answer to that question".
      If the context is , the AI assistant will say, "I'm sorry, but I don't know the answer to that question".
      AI assistant will not apologize for previous responses, but instead will indicated new information was gained.
      AI assistant will not invent anything that is not drawn directly from the context.
      `,
      },
    ]
      let post_prompt = ".Give the information only from the provided CONTEXT. Do not give me any information that are" +
     " not mentioned in the provided CONTEXT."
    messages[messages.length - 1].content = messages[messages.length - 1].content + post_prompt;
    // Ask OpenAI for a streaming chat completion given the prompt
    const response = await openai.createChatCompletion({
      model: 'gpt-3.5-turbo',
      stream: true,
      temperature: 0.2,
      messages: [...prompt, ...messages.filter((message) => message.role === 'user')]
    })
    // Convert the response into a friendly text-stream
    const stream = OpenAIStream(response)
    // Respond with the stream
    let textContent = await processStream(stream);
    const qualifyingDocs = await getQualifyingDocs(lastMessage)
    let url=''
    if(qualifyingDocs.length > 0) {
      url = qualifyingDocs[0].metadata.url;
    }

    return res.send({data:textContent , url: url})
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'An error occurred.' });
  }
});

async function processStream(stream) {
  const chunks = [];
  for await (const chunk of stream) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString('utf-8');
}


const getContext = async (message, namespace, maxTokens = 3000, minScore = 0.7, getOnlyText = true) => {
  // Get the embeddings of the input message
  const embedding = await getEmbeddings(message);

  // Retrieve the matches for the embeddings from the specified namespace
  const matches = await getMatchesFromEmbeddings(embedding, 3, namespace);

  // Filter out the matches that have a score lower than the minimum score
  const qualifyingDocs = matches.filter(m => m.score && m.score > minScore);
  if (!getOnlyText) {
    // Use a map to deduplicate matches by URL
    return qualifyingDocs
  }

  let docs = matches ? qualifyingDocs.map(match => (match.metadata).chunk) : [];
  // Join all the chunks of text together, truncate to the maximum number of tokens, and return the result
  return docs.join("\n").substring(0, maxTokens)
}

const getQualifyingDocs = async (message, namespace, maxTokens = 3000, minScore = 0.7, getOnlyText = true) => {
  // Get the embeddings of the input message
  const embedding = await getEmbeddings(message);

  // Retrieve the matches for the embeddings from the specified namespace
  const matches = await getMatchesFromEmbeddings(embedding, 3, namespace);

  // Filter out the matches that have a score lower than the minimum score
  return matches.filter(m => m.score && m.score > minScore);

}

async function getEmbeddings(input) {
  const val = input.content.replace(/\n/g, ' ')
  try {
    const response = await openai.createEmbedding({
      model: "text-embedding-ada-002",
      input: val
    })

    const result = await response.json();
    return result.data[0].embedding 

  } catch (e) {
    console.log("Error calling OpenAI embedding API: ", e);
    throw new Error(`Error calling OpenAI embedding API: ${e}`);
  }

  
}



const getMatchesFromEmbeddings = async (embeddings, topK, namespace) => {
  // Obtain a client for Pinecone

  const pinecone = new Pinecone({      
    environment: process.env.PINECONE_ENVIRONMENT,      
    apiKey: process.env.PINECONE_API_KEY,      
  });      
     
  const indexName = 'chat-app'|| '';
  if (indexName === '') {
    throw new Error('PINECONE_INDEX environment variable not set')
  }

  // Retrieve the list of indexes to check if expected index exists
  const indexes = await pinecone.listIndexes()
  if (indexes.filter(i => i.name === indexName).length !== 1) {
    throw new Error(`Index ${indexName} does not exist`)
  }

  // Get the Pinecone index
  const index = pinecone.Index(indexName);

  // Get the namespace
  const pineconeNamespace = index.namespace(namespace ?? '')

  try {
    // Query the index with the defined request
    const queryResult = await pineconeNamespace.query({
      vector: embeddings,
      topK,
      includeMetadata: true,
    })


    return queryResult.matches || []
  } catch (e) {
    // Log the error and throw it
    console.log("Error querying embeddings: ", e)
    throw new Error(`Error querying embeddings: ${e}`)
  }
}


app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});

