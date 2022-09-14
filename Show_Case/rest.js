
const getAvailableModels = async () => {
    return await fetch(
        'http://localhost:8081/api/v1/info/models',
        {
            method: "GET",
        })
        .then((response) => response.json())
        .then(data =>  {
            available_models = ["Automatic selection", ...data['models']];
            return available_models;
        })
        .finally(
            document.getElementById("request_text").textContent = "False"
        );
}

const sendFormData = async (request, method, formData) => {
    return await fetch(
        request, 
        {
            method: method,
            body: formData,
        }

        
    )
    .then((response) => response.json())
    .then(data => {
        return data
    })
    .finally(
        document.getElementById("request_text").textContent = "False"
    );
}

const downloadFile = async (type, id) => {
    return await fetch(
        'http://localhost:8081/api/v1/' + type + '/' + id,
        {
            method: 'GET',
            responseType: 'blob',
        }
    ).then((response) => {
        // https://www.delftstack.com/de/howto/javascript/javascript-download/
        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', 'test.pkl');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    })
    .finally(
        document.getElementById("request_text").textContent = "False"
    );
}