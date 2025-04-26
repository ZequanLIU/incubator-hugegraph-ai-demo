# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


from typing import Optional, List

from openai import OpenAI, AsyncOpenAI
import traceback
from hugegraph_llm.utils.log import log


class OpenAIEmbedding:
    def __init__(
            self,
            model_name: str = "text-embedding-3-small",
            api_key: Optional[str] = None,
            api_base: Optional[str] = None
    ):
        api_key = api_key or ''
        if api_key:
            api_key = api_key.strip().replace('\n', '').replace('\r', '')
            log.info(f"Using API key: {api_key[:5]}...{api_key[-5:] if len(api_key) > 10 else ''}")
        
        log.info(f"Initializing OpenAI Embedding with model: {model_name} and API base: {api_base}")
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.aclient = AsyncOpenAI(api_key=api_key, base_url=api_base)
        self.embedding_model_name = model_name

    def get_text_embedding(self, text: str) -> List[float]:
        """Comment"""
        try:
            if text is None:
                log.error("Error: Received None as input text for embedding")
                raise ValueError("输入文本为空（None）。请确保提供有效的查询文本。")
                
            log.info(f"Getting embedding for text: {text[:30] if text else ''}... with model: {self.embedding_model_name}")
            response = self.client.embeddings.create(input=text, model=self.embedding_model_name)
            log.info(f"Successfully got embedding with dimensions: {len(response.data[0].embedding)}")
            return response.data[0].embedding
        except Exception as e:
            log.error(f"Error getting embedding: {e}")
            log.error(traceback.format_exc())
            raise Exception(f"Error getting embedding. Text: '{text[:50] if text else 'None'}...', Model: {self.embedding_model_name}, Error: {e}")

    async def async_get_text_embedding(self, text: str) -> List[float]:
        """Comment"""
        try:
            if text is None:
                log.error("Error: Received None as input text for async embedding")
                raise ValueError("输入文本为空（None）。请确保提供有效的查询文本。")
                
            response = await self.aclient.embeddings.create(input=text, model=self.embedding_model_name)
            return response.data[0].embedding
        except Exception as e:
            log.error(f"Error getting async embedding: {e}")
            log.error(traceback.format_exc())
            raise Exception(f"Error getting async embedding. Text: '{text[:50] if text else 'None'}...', Model: {self.embedding_model_name}, Error: {e}")
