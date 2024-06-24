# for connecting with Bedrock, use Boto3
import boto3
from botocore.config import Config
from openai import AzureOpenAI
import json
import dotenv
import os

# from litellm import completion
from google.oauth2 import service_account
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

# from reka.client import Reka

dotenv.load_dotenv()

class LLM:
    def __init__(
        self,
        model_provider="bedrock/sonnet",  
        system_message="",
    ):
        self.model_provider = model_provider
        # self.model_id = model_id
        self.system_message = system_message

        gcp_json_credentials_dict = json.loads(
            base64.b64decode(os.environ["GCP_BASE64_ENCODED_CREDENTIALS_JSON"])
        )
        credentials = service_account.Credentials.from_service_account_info(
            gcp_json_credentials_dict
        )
        self.vertexai = vertexai.init(
            project="brev-gpu", location="asia-northeast1", credentials=credentials
        )  # limit: 60 calls/minute
        self.init_boto3_clients()

        # self.reka_client = Reka(api_key=os.environ["REKA_API_KEY"])

    def init_boto3_clients(self):
        config = Config(
            region_name="us-east-1",
            connect_timeout=60 * 3,
            read_timeout=60 * 3,
            retries={"max_attempts": 3, "mode": "standard"},
        )

        self.bedrock = boto3.client(service_name="bedrock-runtime", config=config)
        self.bedrock_service = boto3.client(service_name="bedrock", config=config)

        us_west_2_config = Config(
            region_name="us-west-2",
            connect_timeout=60 * 3,
            read_timeout=60 * 3,
            retries={"max_attempts": 3, "mode": "standard"},
        )
        # opus only available here
        self.us_west_2_bedrock = boto3.client(
            service_name="bedrock-runtime", config=us_west_2_config
        )

    def generate_response(self, messages):
        if self.model_provider == "bedrock/sonnet":
            body = json.dumps(
                {
                    "system": (
                        self.system_message if len(self.system_message) > 0 else ""
                    ),
                    "anthropic_version": "bedrock-2023-05-31",
                    "temperature": 0,
                    "max_tokens": 200000,
                    "messages": messages,  # Bedrock DOES NOT support System message as first item in messages arr, express as system key
                }
            )
            response = self.bedrock.invoke_model(
                body=body,
                # modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())
            return response_body.get("content")[0]["text"]
        elif self.model_provider == "bedrock/haiku":
            body = json.dumps(
                {
                    "system": (
                        self.system_message if len(self.system_message) > 0 else ""
                    ),
                    "anthropic_version": "bedrock-2023-05-31",
                    "temperature": 0,
                    "max_tokens": 200000,
                    "messages": messages,
                }
            )
            response = self.bedrock.invoke_model(
                body=body,
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())
            return response_body.get("content")[0]["text"]
        elif self.model_provider == "bedrock/opus":
            body = json.dumps(
                {
                    "system": (
                        self.system_message if len(self.system_message) > 0 else ""
                    ),
                    "anthropic_version": "bedrock-2023-05-31",
                    "temperature": 0,
                    "max_tokens": 200000,
                    "messages": messages,
                }
            )
            response = self.us_west_2_bedrock.invoke_model(
                body=body,
                modelId="anthropic.claude-3-opus-20240229-v1:0",
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())
            return response_body.get("content")[0]["text"]
        elif self.model_provider == "bedrock/commandr+":
            body = json.dumps(
                {
                    "preamble": (
                        self.system_message if len(self.system_message) > 0 else ""
                    ),
                    "chat_history": messages[:-1] if len(messages) > 0 else [],
                    "message": messages[-1]["content"],
                }
            )
            response = self.us_west_2_bedrock.invoke_model(
                body=body,
                modelId="cohere.command-r-plus-v1:0",
                accept="*/*",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())
            # print("CommandR+ response body: ", response_body)
            return response_body["text"]
        elif self.model_provider == "bedrock/commandr":
            body = json.dumps(
                {
                    "preamble": (
                        self.system_message if len(self.system_message) > 0 else ""
                    ),
                    "chat_history": messages[:-1] if len(messages) > 0 else [],
                    "message": messages[-1]["content"],
                }
            )
            response = self.us_west_2_bedrock.invoke_model(
                body=body,
                modelId="cohere.command-r-v1:0",
                accept="*/*",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())
            # print("CommandR+ response body: ", response_body)
            return response_body["text"]
        # elif self.model_provider == "bedrock/llama-70b":
        #     body = json.dumps(
        #         {
        #             "prompt": "this is where you place your input text",
        #             "max_gen_len": 512,
        #             "temperature": 0.5,
        #             "top_p": 0.9,
        #         },
        #     )
        #     response = bedrock.invoke_model(
        #         body=body,
        #         modelId="meta.llama3-70b-instruct-v1:0",
        #         accept="application/json",
        #         contentType="application/json",
        #     )
        #     return response_body["text"]
        elif self.model_provider == "azure/gpt-4t-128k":
            client = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_GPT4T_BASE_URL"],
                api_version="2024-02-15-preview",
                api_key=os.environ["AZURE_GPT4T_API_KEY"],
            )

            response = client.chat.completions.create(
                model="turbo-128k",
                temperature=0,
                messages=messages,
            )
            res = response.choices[0].message.content
            return res
        elif self.model_provider == "azure/gpt-3.5t":
            client = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_BASE_URL"],
                api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
            )

            response = client.chat.completions.create(
                model=os.environ["AZURE_OPENAI_MODEL_ID"],
                temperature=0,
                messages=messages,
            )
            res = response.choices[0].message.content
            return res
        # elif self.model_provider == "reka/core":
        #     response = self.reka_client.chat.create(
        #         messages=messages,
        #         model="reka-core-20240501",
        #     )
        #     return response.responses[0].message.content
        elif self.model_provider == "gemini/1.5-pro":
            generation_config = {
                "max_output_tokens": 8192,
                "temperature": 1,
                "top_p": 0.95,
            }
            safety_settings = {
                generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            model = GenerativeModel("gemini-1.5-pro-001")
            responses = model.generate_content(
                [messages[0]["content"]],
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=True,
            )

            final_answer = ""
            for response in responses:
                final_answer += response.text
            return final_answer
        # elif self.model_provider == "gemini/1.5-pro":
        #     genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        #     generation_config = {
        #     "temperature": 1,
        #     "top_p": 0.95,
        #     "top_k": 64,
        #     "max_output_tokens": 8192,
        #     "response_mime_type": "text/plain",
        #     }
        #     safety_settings = [
        #     {
        #         "category": "HARM_CATEGORY_HARASSMENT",
        #         "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        #     },
        #     {
        #         "category": "HARM_CATEGORY_HATE_SPEECH",
        #         "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        #     },
        #     {
        #         "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        #         "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        #     },
        #     {
        #         "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        #         "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        #     },
        #     ]

        #     model = genai.GenerativeModel(
        #     model_name="gemini-1.5-pro-latest",
        #     safety_settings=safety_settings,
        #     generation_config=generation_config,
        #     )

        #     chat_session = model.start_chat(
        #     history=[
        #     ]
        #     )
        #     message = messages[0]["content"]
        #     response = chat_session.send_message(message)
        #     return response.text
        # elif self.model_provider == "litellm":
        #     response = completion(model=self.model_id, messages=messages, stream=True)

        #     history = ""
        #     for chunk in response:
        #         content = chunk["choices"][0]["delta"]["content"]
        #         if content is not None:
        #         history += content
        #         yield history


# def call_litellm(input_text):
# #     client = OpenAI(
# #     base_url="http://0.0.0.0:4000"
# # )
#     messages=[
#         {"role": "system", "content": "You are an expert summarizer"},
#         {"role": "user", "content": input_text}
#         ]

#     # response = client.chat.completions.create(model="claude-3", messages=messages, stream=True)
#     response = completion(model="claude-3-opus-20240229", messages=messages, stream=True)

#     # return response.choices[0].message.content
#     history = ""
#     for chunk in response:
#       content = chunk["choices"][0]["delta"]["content"]
#       if content is not None:
#         history += content
#       yield history
