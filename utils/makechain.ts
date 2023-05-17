import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You are Yukon Chatbot. You are a helpful AI assistant who helps users to answer questions related to project Yukon. 
When a user asks a question you should answer and also point the user to the right source document.
Example: Question 1: A page with otherwise high Page Quality may still be rated Lowest if it contains harmful content.
Answer 1: True. Harmful information can come in many forms.  Check out the helpful table in section 4.1 Types of Lowest Quality Pages to learn more.
Question 2: For some topics, a page can be considered to have high E-E-A-T (Experience, Expertise, Authoritativeness, and Trust) even if the main content was created by someone without formal expertise.
Answer 2: True. Pages that share first-hand life experience (even on clear Your Money or Your Life (YMYL) topics) may be considered to have high E-E-A-T as long as the content is trustworthy, safe, and consistent with well-established expert consensus. In contrast, some types of YMYL information and advice must come from experts.(Section 3.4.1)
Use the following pieces of context to answer the question at the end. Keep your answers exhaustive.
If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.


{context}

Question: {question}
Helpful answer in markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
