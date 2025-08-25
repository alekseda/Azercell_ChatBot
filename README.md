
## Homework 3 — Building and Deploying a RAG-based AI Chatbot
 This project builds upon Homework 1’s AWS Bedrock-based chatbot by separating it into a FastAPI backend and a Streamlit frontend. Both services are containerized with Docker, orchestrated using Docker Compose, and deployed to an AWS EC2 instance through a CI/CD pipeline powered by a self-hosted GitHub Actions runner. The backend integrates AWS Bedrock’s Claude 3.5 Sonnet model for natural language reasoning, while the frontend delivers an interactive web interface for querying the chatbot and visualizing its responses.

## The project structure highlights the clear separation between backend and frontend services:
<img width="480" height="422" alt="image" src="https://github.com/user-attachments/assets/e3f3ee67-c3b7-4063-8bca-fb9fd7017b0c" />

## Prerequisites

- AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- Docker and Docker Compose (for containerization)
    - AWS Account with access to:  
    - Bedrock (Claude 3.5 Sonnet model: anthropic.claude-3-5-sonnet-20240620-v1:0)
- Knowledge base (ID: JGMPKF6VEI)
- EC2 instance (for deployment)
- GitHub Repository with SSH access configured
- Python 3.9+ (for local development)
- GitHub Actions Self-Hosted Runner (set up on EC2)

## Setup Guide
### Running Locally

  1. Clone the Repository
     ```
     git clone <your-repo-url>
     cd <your-repo-name>

  2. Configure Environment Variables

      Copy the example .env files for both backend and frontend:
      ```
      cp backend/.env.example backend/.env
      cp frontend/.env.example frontend/.env


  3. Update the backend .env with your AWS credentials and settings:
      ```
      AWS_ACCESS_KEY_ID=<your-access-key>
      AWS_SECRET_ACCESS_KEY=<your-secret-key>
      AWS_REGION=us-east-1
      KNOWLEDGE_BASE_ID=JGMPKF6VEI
      CLAUDE_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0


  4. Point the frontend to the backend by updating frontend/.env:
      ```
      BACKEND_URL=http://backend:8000

## Build and Launch

  1. Use Docker Compose to build and start both services:
      ```
      docker-compose up --build 
      ```
        * FastAPI backend will be accessible at http://localhost:8000/docs
        * Streamlit frontend can be accessed at http://localhost:8501

  ## Test the Application

  Open the frontend in your browser: ` http://localhost:8501`
  Enter a query (for example, "Hello, can u help me?").

