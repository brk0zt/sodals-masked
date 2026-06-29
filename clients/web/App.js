// src/App.js - SECURE CLIENT

import React, { useState, useEffect, useRef } from 'react';
import useWebSocket from 'react-use-websocket';
// SHA256 için crypto-js gereklidir. Terminalde: npm install crypto-js
import CryptoJS from 'crypto-js'; 

const WS_URL = 'ws://localhost:5500/ws';
const SECRET_API_KEY = "SECRET_API_KEY"; // Sunucudaki ile AYNI olmalı

function App() {
  const [socketUrl, setSocketUrl] = useState(WS_URL);
  const [messageHistory, setMessageHistory] = useState([]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("DISCONNECTED");
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  
  const { sendMessage, lastMessage, readyState } = useWebSocket(socketUrl, {
    onOpen: () => {
      console.log('WS Bağlandı, Kimlik Doğrulama Bekleniyor...');
      setStatus("HANDSHAKING...");
    },
    onClose: () => {
      console.log('WS Kapandı');
      setStatus("DISCONNECTED");
      setIsAuthenticated(false);
    },
    shouldReconnect: (closeEvent) => true,
  });

  useEffect(() => {
    if (lastMessage !== null) {
      try {
        const data = JSON.parse(lastMessage.data);
        
        // --- 1. GÜVENLİK EL SIKIŞMASI (HANDSHAKE) ---
        if (data.AuthChallenge) {
            console.log(" Meydan Okuma Alındı:", data.AuthChallenge.nonce);
            
            // Token Hesapla: SHA256(nonce + SECRET)
            const nonce = data.AuthChallenge.nonce;
            const rawString = nonce + SECRET_API_KEY;
            const token = CryptoJS.SHA256(rawString).toString(CryptoJS.enc.Hex);
            
            // Cevabı Gönder
            const responseMsg = {
                AuthResponse: {
                    token: token
                }
            };
            sendMessage(JSON.stringify(responseMsg));
            console.log(" Token Gönderildi:", token);
        }
        
        else if (data.type === "AuthSuccess") {
            console.log(" Kimlik Doğrulama BAŞARILI!");
            setIsAuthenticated(true);
            setStatus("CONNECTED & SECURE");
            
            // Kimlik Tanıtımı (Opsiyonel)
            sendMessage(JSON.stringify({ 
                user_id: "web_admin_01",
                type: "auth" 
            }));
        }
        
        else if (data.type === "AuthFailed") {
            console.error(" Kimlik Doğrulama BAŞARISIZ!");
            setStatus("ACCESS DENIED");
            // Soket otomatik kapanacaktır.
        }
        
        // --- NORMAL MESAJ AKIŞI ---
        else if (isAuthenticated) {
            // Dashboard güncellemesi veya LlmResponse
            if (data.DashboardUpdate) {
                // Dashboard verilerini işle...
                // (Buraya eski dashboard kodlarını entegre edebilirsin)
            } else {
                setMessageHistory((prev) => prev.concat(lastMessage));
            }
        }
      } catch (e) {
        // Text mesajı olabilir
        console.log("Raw Message:", lastMessage.data);
      }
    }
  }, [lastMessage, sendMessage, isAuthenticated]);

  const handleSend = () => {
    if (!isAuthenticated) {
        alert("Bağlantı güvenli değil!");
        return;
    }
    sendMessage(JSON.stringify({ 
        user_id: "web_admin_01",
        text: input,
        image: null 
    }));
    setInput("");
  };

  return (
    <div style={{ padding: 20, fontFamily: 'monospace', backgroundColor: '#1e1e1e', color: '#0f0', minHeight: '100vh' }}>
      <h1>SODALS SECURE TERMINAL</h1>
      <div style={{ border: '1px solid #333', padding: 10, marginBottom: 10 }}>
        STATUS: <span style={{ color: isAuthenticated ? '#0f0' : '#f00' }}>{status}</span>
      </div>

      <div style={{ height: 400, overflowY: 'scroll', border: '1px solid #333', padding: 10, marginBottom: 10 }}>
        {messageHistory.map((message, idx) => (
          <div key={idx}>{message.data}</div>
        ))}
      </div>

      <div style={{ display: 'flex' }}>
        <input
          style={{ flex: 1, padding: 10, backgroundColor: '#333', color: '#fff', border: 'none' }}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          placeholder={isAuthenticated ? "Komut girin..." : "Bağlantı bekleniyor..."}
          disabled={!isAuthenticated}
        />
        <button onClick={handleSend} disabled={!isAuthenticated} style={{ padding: 10, backgroundColor: '#005500', color: '#fff', border: 'none', cursor: 'pointer' }}>
          GÖNDER
        </button>
      </div>
    </div>
  );
}

export default App;