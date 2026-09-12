function handleForm (e){
    e.preventDefault();
}

async function getChatResponse (){
    const response = await fetch('/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt })
    });
}