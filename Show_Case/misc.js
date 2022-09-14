async function getFile() {
    var fileupload = document.getElementById("fileupload");
    var separator_input = document.getElementById("separator");

    file = fileupload.files[0];
    separator = separator_input.value;


    return [file, separator];
}

async function updateOutput() {
    document.getElementById("file_id_text").value = file_id
    document.getElementById("seasonality_text").value = seasonality
    document.getElementById("model_id_text").value = model_id
    document.getElementById("prediction_text").value = prediction
    document.getElementById("model_rmse_text").value = rmse_value
    document.getElementById("model_type_text").value = model_type
}