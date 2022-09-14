
let file_id = -1;
let seasonality = null;
let model_id = -1;
let prediction = []
let rmse_value = -1
let model_type = null

window.onload = async function ()  {

    var model_selection = document.getElementById("modeltype");

    var available_models = await getAvailableModels();


    for (var i = 0; i < available_models.length; i++) {
        var opt = available_models[i];
        var el = document.createElement("option");
        el.textContent = opt;
        el.value = opt;
        model_selection.appendChild(el);
    }

}

async function uploadFile() {
    [file, separator] = await getFile();

    let formData = new FormData();
    if (file == undefined) {
        alert("Please select a file")
        return
    }

    formData.append("data", file);
    formData.append("separator", separator);
    data = await sendFormData('http://localhost:8081/api/v1/files', 'PUT', formData);
    result = data['file_id']

    if (Number.isInteger(result)) {
        file_id = result;
    }
    updateOutput()
}

async function analyseFile() {
    var interval = document.getElementById("time_interval").value

    allowed_intervals = ['S', 'min', 'H', 'D', 'W', 'M', 'Q', 'A']

    if (!allowed_intervals.includes(interval)) {
        alert("Only the intervals " + allowed_intervals + " are allowed. Please fill in the interval field")
        return
    }

    requestURL = 'http://localhost:8081/api/v1/data/analyse'
    let formData = new FormData();

    if (file_id == -1) {
        [file, separator] = await getFile();
        if (file == undefined) {
            alert("Please select a file")
            return
        }

        formData.append("data", file);
        formData.append("separator", separator);
    } else {
        requestURL = requestURL + "/" + file_id
    }

    formData.append('interval', interval)
    
    response = await sendFormData(requestURL, "POST", formData)
    file_id = response['file_id']
    seasonality = response['seasonality']
    updateOutput()
}

async function trainModel() {
    var selected_model = document.getElementById("modeltype").value

    requestURL = 'http://localhost:8081/api/v1/models'
    let formData = new FormData();

    if (file_id == -1) {
        alert("Please upload a file fisrt")
        return
    } else {
        formData.append('data_id', file_id);
    }

    if (selected_model == "Automatic selection") {
        await autoSelectModel()
        return
    }

    formData.append('model_type', selected_model)
    
    response = await sendFormData(requestURL, "POST", formData)
    model_id = response['model_id']
    rmse_value = response['errors']['rmse']
    model_type = selected_model
    updateOutput()
}

async function autoSelectModel() {
    requestURL = 'http://localhost:8081/api/v1/auto/' + file_id
    let formData = new FormData();
    var interval = document.getElementById("time_interval").value
    var prediciton_steps = document.getElementById("prediction_amount").value

    if (prediciton_steps == "" || prediciton_steps < 1) {
        alert("Automatik model Selection without a prediction.")
        prediciton_steps = 0
    }

    allowed_intervals = ['S', 'min', 'H', 'D', 'W', 'M', 'Q', 'A']

    if (!allowed_intervals.includes(interval)) {
        alert("Only the intervals " + allowed_intervals + " are allowed. Please fill in the interval field")
        return
    }

    formData.append('prediction_interval', prediciton_steps)
    formData.append('interval', interval)

    response = await sendFormData(requestURL, "POST", formData)

    model_id = response['final_model']['model_id']
    rmse_value = response['final_model']['errors']['rmse']
    model_type = response['final_model']['model_type']

    updateOutput()
    return
}

async function downloadModel() {
    downloadFile('models', model_id)
}

async function predict() {
    if (model_id == -1) {
        alert("No model has been trained")
        return
    }

    requestURL = 'http://localhost:8081/api/v1/models/' + model_id + "/prediction"

    let formData = new FormData();

    var prediciton_steps = document.getElementById("prediction_amount").value

    if (prediciton_steps == "" || prediciton_steps < 1) {
        alert("Please insert a number larger than 0 as the amount of prediction steps")
        return
    }

    formData.append('prediction_interval', prediciton_steps)

    response = await sendFormData(requestURL, "PUT", formData)
    prediction = response['prediction']
    updateOutput()
}
