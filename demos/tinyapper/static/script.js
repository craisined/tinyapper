const socket = io();

function handleForm(e) {
    e.preventDefault();
    const formData = new FormData(form);
    const data = Object.fromEntries(formData);
    socket.emit("chat", {prompt: data.prompt})
}
document.querySelector(".input-container").addEventListener(handleForm);

socket.on("output", (data) => {
    console.log("The server sent some news:", data);
});
