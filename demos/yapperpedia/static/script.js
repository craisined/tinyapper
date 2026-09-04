// AI generated and modified
async function readStream() {
    const response = await fetch('/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const outputElement = document.querySelector('.article-content-container');
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });

        // 2. Extract complete SSE frames separated by double newlines
        const lines = buffer.split('\n\n');
        buffer = lines.pop(); // Keep partial trailing chunk in buffer

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const rawPayload = line.slice(6);
                if (rawPayload.trim() === '[DONE]') break;
                const fullHTML = rawPayload.replace(/\\n/g, '\n');
                outputElement.innerHTML = fullHTML;
            }
        }
    }
}

readStream()