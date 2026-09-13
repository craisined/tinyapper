const socket = io()

const form = document.querySelector(".input-container")
function handleForm(e) {
    e.preventDefault();
    const formData = new FormData(form);
    const data = Object.fromEntries(formData);
    socket.emit("chat", {prompt: data.prompt})
}

form.addEventListener('submit', handleForm);

socket.on("output", (data) => {
    console.log("The server sent some news:", data);
});
