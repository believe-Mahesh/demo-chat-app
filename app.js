const { Pinecone, ScoredPineconeRecord } = require("@pinecone-database/pinecone");
const express = require('express');
const bodyParser = require('body-parser');
const { Configuration, OpenAIApi, ChatCompletionRequestMessageRoleEnum } = require('openai-edge');
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
        role: ChatCompletionRequestMessageRoleEnum.System,
        content: `You are a empathetic answering assitant that can comply with any questions of the New Mexico citizens.
        Your traits include empathy, kindness, helpfulness, cleverness and articulateness.
        You will greet user automatically as "Hello there! I am New Mexico Citizen Services Bot. Can you please tell me your name ?".
        You are a well behaved and well mannered individual.
        You must assume that the user is a resident of the New Mexico state in USA and Answer all the questions empathetically 
        and personally by addressing him with his first name.
        START OF CONTEXT BLOCK
        ${context}
        END OF CONTEXT BLOCK
        You will take into account any CONTEXT BLOCK that is provided in a conversation.
        If the context does not provide the answer to question, the You must say, "I'm sorry, but I don't know the answer to that question".
        If the context is , the You must say, "I'm sorry, but I don't know the answer to that question".
        You must not invent anything that is not drawn directly from the context.
        You always answer the with markdown formatting. You will be penalized if you do not answer with markdown when it would be possible.
        The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes.
        You do not support images and never include images. You will be penalized if you render images.
        `,
      },
    ]
    let post_prompt = ".Give the information only from the provided Context Block. You always answer the with markdown formatting. You will be penalized if you do not answer with markdown when it would be possible. The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes."
    messages[messages.length - 1].content = messages[messages.length - 1].content + post_prompt;
    // Ask OpenAI for a streaming chat completion given the prompt
    const response = await openai.createChatCompletion({
      model: 'gpt-3.5-turbo',
      stream: true,
      temperature: 0.2,
      messages: [...prompt, ...messages]
    })
    // Convert the response into a friendly text-stream
    const stream = OpenAIStream(response);
    // Respond with the stream
    //let textContent = await processStream(stream);

    // const qualifyingDocs = await getQualifyingDocs(lastMessage)
    // let url=''
    // if(qualifyingDocs.length > 0) {
    //   url = qualifyingDocs[0].metadata.url;
    // }
    //console.log('the response is ', response);
    for await (const chunk of stream) {
      //console.log(new TextDecoder().decode(chunk) || '')
      res.write(new TextDecoder().decode(chunk))
    }
    res.end();
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'An error occurred.' });
  }
});

app.post('/api/urls', async (req, res) => {
  const { messages } = req.body;
  const urls = [];
  // Get the last message
  const lastMessage = messages[messages.length - 1];
  const qualifyingDocs = await getQualifyingDocs(lastMessage, '');
  if (qualifyingDocs.length > 0) {
    qualifyingDocs.forEach((doc) => {
      urls.push(doc.metadata?.url);
    })
  }
  return res.send({ 'url': urls });
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

const getQualifyingDocs = async (message, namespace, maxTokens = 3000, minScore = 0.8, getOnlyText = true) => {
  // Get the embeddings of the input message
  const embedding = await getEmbeddings(message);

  // Retrieve the matches for the embeddings from the specified namespace
  const matches = await getMatchesFromEmbeddings(embedding, 2, namespace);

  // Filter out the matches that have a score lower than the minimum score
  return matches.filter(m => m.score && m.score >= minScore);

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

  const indexName = 'chat-app' || '';
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

