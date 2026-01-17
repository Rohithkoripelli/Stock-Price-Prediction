# Chat Interface Setup Guide

## Overview
The chat interface allows users to ask questions about stock predictions in natural language using OpenAI GPT-4o.

## Features
- Natural language queries about stock predictions
- Real-time responses powered by GPT-4o
- Context-aware answers based on current prediction data
- Example questions for easy interaction

## Setup Instructions

### 1. Get an OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the API key (starts with `sk-...`)

### 2. Local Development Setup

Create a file `web/.env.local` with your API key:

```bash
cd web
cp .env.local.example .env.local
```

Edit `.env.local` and add your key:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 3. Vercel Deployment Setup

To deploy with the chat feature enabled:

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Select your project: `web`
3. Go to Settings → Environment Variables
4. Add a new variable:
   - **Name**: `OPENAI_API_KEY`
   - **Value**: Your OpenAI API key (starts with `sk-...`)
   - **Environment**: Production, Preview, Development (select all)
5. Click "Save"
6. Redeploy your application

### 4. Deploy to Vercel

From the `web` directory:
```bash
vercel --prod --yes
```

Or from the root directory:
```bash
cd web && vercel --prod --yes
```

## Example Questions Users Can Ask

- "What's the prediction for HDFC Bank?"
- "Which stocks are strong buy?"
- "What's the best opportunity today?"
- "Should I buy ICICI Bank?"
- "Compare SBI and HDFC Bank"
- "Which stocks are predicted to go down?"
- "What are the confidence levels for each stock?"

## How It Works

1. User types a question in natural language
2. The question is sent to `/api/chat` endpoint
3. The API loads current prediction data
4. OpenAI GPT-4o analyzes the question with prediction context
5. AI generates a helpful, context-aware response
6. Response is displayed in the chat interface

## Security Notes

- Never commit `.env.local` to git (it's in .gitignore)
- Keep your OpenAI API key secret
- Set usage limits in OpenAI dashboard to control costs
- The API key is only used server-side, never exposed to clients

## Cost Considerations

- GPT-4o pricing: ~$0.005 per 1K input tokens, ~$0.015 per 1K output tokens
- Average chat response: ~500-1000 tokens total
- Estimated cost: $0.01-0.02 per conversation
- Set monthly budget limits in OpenAI dashboard

## Troubleshooting

**Chat not working locally:**
- Verify `.env.local` exists in the `web` directory
- Check API key is correct (starts with `sk-`)
- Restart the dev server after adding env variables

**Chat not working on Vercel:**
- Verify environment variable is set in Vercel dashboard
- Redeploy after adding the variable
- Check deployment logs for errors

**"OpenAI API key not configured" error:**
- Environment variable not set correctly
- Variable name must be exactly `OPENAI_API_KEY`
- Redeploy after setting the variable
