const socket = io()

function create_chatbox(author, msg) {

    // AI generated
    const container = document.createElement("div");
    container.className = "message-container";
    const bTag = document.createElement("b");
    bTag.textContent = author;
    const brTag = document.createElement("br");
    const spanTag = document.createElement("span");
    spanTag.textContent = msg;
    container.append(bTag, brTag, spanTag);
    const trailingBr = document.createElement("br");

    const chatContainer = document.querySelector(".chat-container");
    chatContainer.append(container, trailingBr);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function handleForm(e) {
    e.preventDefault();
    const formData = new FormData(e.target);
    e.target.reset()
    const data = Object.fromEntries(formData);
    create_chatbox("user", data.prompt);
    socket.emit("chat", { prompt: data.prompt })
}
document.querySelector(".input-container").addEventListener('submit', handleForm);

socket.on("output", (data) => {
    const newChat = document.createElement("p");
    create_chatbox("tinyapper", data.msg);
});

socket.on("disconnect", (data) => {
    setTimeout(() => {
        window.location.reload();
    }, 1000);
});
