use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio_stream::Stream;
use std::pin::Pin;

const DEFAULT_BASE: &str = "http://127.0.0.1:22434";

#[derive(Clone)]
pub struct MeticuloApi {
    client: Client,
    base_url: String,
}

#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatStreamChunk {
    pub message: Option<ChatMessageContent>,
    pub done: bool,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessageContent {
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub modified_at: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelList {
    pub models: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
pub struct GenerateResponse {
    pub response: Option<String>,
    pub done: bool,
}

#[derive(Debug, Deserialize)]
pub struct HealthResponse {
    pub status: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct PsModel {
    pub name: String,
    #[serde(default)]
    pub size: u64,
}

#[derive(Debug, Deserialize)]
pub struct PsResponse {
    #[serde(default)]
    pub models: Vec<PsModel>,
}

#[derive(Debug, Deserialize)]
pub struct CatalogResponse {
    #[serde(default)]
    pub models: Vec<String>,
}

impl MeticuloApi {
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .pool_max_idle_per_host(2)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: DEFAULT_BASE.to_string(),
        }
    }

    pub async fn health(&self) -> Result<HealthResponse, String> {
        let url = format!("{}/api/health", self.base_url);
        let resp = self.client.get(&url).send().await.map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("Health check failed: {}", resp.status()));
        }
        resp.json::<HealthResponse>().await.map_err(|e| e.to_string())
    }

    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, String> {
        let url = format!("{}/api/tags", self.base_url);
        let resp = self.client.get(&url).send().await.map_err(|e| e.to_string())?;
        let list: ModelList = resp.json().await.map_err(|e| e.to_string())?;
        Ok(list.models)
    }

    pub async fn list_running(&self) -> Result<Vec<PsModel>, String> {
        let url = format!("{}/api/ps", self.base_url);
        let resp = self.client.get(&url).send().await.map_err(|e| e.to_string())?;
        let ps: PsResponse = resp.json().await.map_err(|e| e.to_string())?;
        Ok(ps.models)
    }

    pub async fn fetch_catalog(&self) -> Result<Vec<String>, String> {
        let url = format!("{}/api/catalog", self.base_url);
        let resp = self.client.get(&url).send().await.map_err(|e| e.to_string())?;
        let cat: CatalogResponse = resp.json().await.map_err(|e| e.to_string())?;
        Ok(cat.models)
    }

    pub fn chat_stream(
        &self,
        model: &str,
        messages: &[ChatMessage],
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<u32>,
    ) -> Pin<Box<dyn Stream<Item = Result<String, String>> + Send + '_>> {
        let url = format!("{}/api/chat", self.base_url);
        let req = ChatRequest {
            model: model.to_string(),
            messages: messages.to_vec(),
            stream: true,
            temperature,
            top_p,
            top_k,
        };
        let client = self.client.clone();

        Box::pin(async_stream::stream! {
            use futures::StreamExt;

            let resp = match client.post(&url).json(&req).send().await {
                Ok(r) => r,
                Err(e) => {
                    yield Err(format!("Connection failed: {e}"));
                    return;
                }
            };

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                yield Err(format!("Server error {status}: {body}"));
                return;
            }

            let mut byte_stream = resp.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk_result) = byte_stream.next().await {
                match chunk_result {
                    Ok(bytes_data) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes_data));
                        // Process complete JSON lines
                        while let Some(nl) = buffer.find('\n') {
                            let line = buffer[..nl].trim().to_string();
                            buffer = buffer[nl + 1..].to_string();

                            if line.is_empty() {
                                continue;
                            }

                            match serde_json::from_str::<ChatStreamChunk>(&line) {
                                Ok(parsed) => {
                                    if let Some(msg) = parsed.message {
                                        if !msg.content.is_empty() {
                                            yield Ok(msg.content);
                                        }
                                    }
                                    if parsed.done {
                                        return;
                                    }
                                }
                                Err(_) => {
                                    // Skip malformed chunks
                                }
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(format!("Stream error: {e}"));
                        return;
                    }
                }
            }
        })
    }

    pub async fn pull_model(&self, name: &str) -> Result<(), String> {
        let url = format!("{}/api/pull", self.base_url);
        let body = serde_json::json!({ "model": name });
        let resp = self.client.post(&url).json(&body).send().await.map_err(|e| e.to_string())?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("Pull failed: {}", resp.status()))
        }
    }

    pub async fn load_model(&self, name: &str) -> Result<(), String> {
        let url = format!("{}/api/load", self.base_url);
        let body = serde_json::json!({ "model": name });
        let resp = self.client.post(&url).json(&body).send().await.map_err(|e| e.to_string())?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("Load failed: {}", resp.status()))
        }
    }

    pub async fn warmup_model(&self, name: &str) -> Result<(), String> {
        let url = format!("{}/api/generate", self.base_url);
        let body = serde_json::json!({
            "model": name,
            "prompt": "",
            "stream": false,
            "keep_alive": -1
        });
        let resp = self.client.post(&url).json(&body).send().await.map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("Warmup failed: {}", resp.status()));
        }
        let payload: GenerateResponse = resp.json().await.map_err(|e| e.to_string())?;
        let _response_chars = payload.response.as_deref().map(|s| s.len()).unwrap_or(0);
        if payload.done {
            Ok(())
        } else {
            Err("Warmup did not complete".into())
        }
    }

    pub async fn delete_model(&self, name: &str) -> Result<(), String> {
        let url = format!("{}/api/delete", self.base_url);
        let body = serde_json::json!({ "name": name });
        let resp = self.client.delete(&url).json(&body).send().await.map_err(|e| e.to_string())?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("Delete failed: {}", resp.status()))
        }
    }
}
