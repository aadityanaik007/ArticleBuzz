
class APICaller{
    static csv_data = {}
    static prediction_data = {"article_name":[],"shares":[]}
    static metrics_data = {"xg_boost":{"MAE":0,"RMS":0},"gradient":{"MAE":0,"RMS":0},"some_different_model":{"MAE":0,"RMS":0}}
    static models = {}
    static model_list = []
    static RMS_list = []
    static MAE_list = []
    constructor(){
        console.log("Called");
    }

    async uploadCsv() {
        const fileInput = document.getElementById('fileInput');
        // const selectedFiles = fileInput.files;

        if (fileInput) {
            const selectedFile = fileInput.files[0];

            if (selectedFile) {
                var formdata = new FormData();
                formdata.append("file", selectedFile);

                var requestOptions = {
                    method: 'POST',
                    body: formdata,
                    redirect: 'follow'
                };

                APICaller.csv_data = await fetch("http://127.0.0.1:8000/get_prediction_for_csv", requestOptions)
                    .then(response => response.text())
                    .then(result => {
                        result =  JSON.parse(result)
                        result['model'].forEach((model)=>{
                            APICaller.model_list.push(Object.keys(model)[0])
                            const model_name = Object.keys(model)[0]
                            console.log(model_name);
                            APICaller.RMS_list.push(model[model_name]['RMS'])
                            // console.log(model[model_name]['RMS']);
                            APICaller.MAE_list.push(model[model_name]['MAE'])
                        })
                        result['data'].forEach((item)=>{
                            APICaller.prediction_data['article_name'].push(item['article_name'])
                            APICaller.prediction_data['shares'].push(item['number_of_shares'])
                        })
                    })
                    .catch(error => console.log('error', error));
                document.getElementById('upload_csv_label').textContent = "PREDICTIONS ARE AVAILABLE CLICK ON THE SHOW GRAPH TAB"
            } else {
                console.error('No file selected.');
            }
        } else {
            console.error('File input element not found.');
        }

    }
    showGraph(){
        console.log(APICaller.prediction_data);
        const elementToRemove = document.getElementById('upload_csv_label')
        if (elementToRemove) {
            elementToRemove.remove();
        } else {
            console.error('Element not found');
        }
        this.share_predict_bar(APICaller.prediction_data)
        this.mertric_for_graph(APICaller.prediction_data)
        // APICaller.csv_data.forEach((item) => {

        // }
        // const data = APICaller.csv_data
        // .then((result)=>{return result.data})
        
        // console.log(data);
        // return csv_data
    }

    share_predict_bar(dataset){
        const graphContainer = document.getElementById('graph');
        // const canvas = document.createElement('canvas')
        // canvas.setAttribute('id','prediction_result')
        // graphContainer.appendChild(canvas)
        const canvas = document.getElementById('prediction_result');
        canvas.width = graphContainer.clientWidth;
        canvas.height = graphContainer.clientHeight;
        
        new Chart(canvas, {
            type: 'bar',
            data: {
            labels: dataset['article_name'],
            datasets: [{
                label: 'Number of Articles Shares',
                data: dataset['shares'],
                borderWidth: 1
            }]
            },
            options: {
            maintainAspectRatio: false, // Disable the aspect ratio constraint
            responsive: true, 
            scales: {
                y: {
                beginAtZero: true
                }
            }
            }
        });
    }

    mertric_for_graph(){
        let graphContainer = document.getElementById('myDiv');
        graphContainer.setAttribute("style","margin-left: 150px;margin-top: 50px;width: 80%;padding: 10px;min-height: 500px;height: auto;border: #12E1B9 solid 1px;color: #12E1B9;")

        var xValue = APICaller.model_list;
        console.log(xValue);
        var yValue = APICaller.MAE_list;
        console.log(yValue);
        var yValue2 = APICaller.RMS_list;
        console.log(yValue2);

        var testMAE = {
        x: xValue,
        y: yValue,
        type: 'bar',
        text: yValue.map(String),
        textposition: 'auto',
        hoverinfo: 'none',
        opacity: 0.5,
        name: 'MAE',
        marker: {
            color: 'rgb(158,202,225)',
            line: {
            color: 'rgb(8,48,107)',
            width: 1.5
            }
        }
        };

        var testRMSE = {
        x: xValue,
        y: yValue2,
        type: 'bar',
        text: yValue2.map(String),
        textposition: 'auto',
        hoverinfo: 'none',
        name: 'RMSE',
        marker: {
            color: 'rgba(58,200,225,.5)',
            line: {
            color: 'rgb(8,48,107)',
            width: 1.5
            }
        }
        };

        var data = [testMAE,testRMSE];

        var layout = {
        title: 'Performance of different Machine Learning models'
        };

        Plotly.newPlot('myDiv', data, layout);

    }
}

function scrollToBottom(){
    var bottomDiv = document.getElementById('graph');
    bottomDiv.scrollIntoView({ behavior: 'smooth' });
}