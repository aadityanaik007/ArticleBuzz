const form = document.querySelector("form"),
fileInput = document.querySelector(".file-input"),
progressArea = document.querySelector(".progress-area"),
uploadedArea = document.querySelector(".uploaded-area");

form.addEventListener("click", () =>{
  fileInput.click();
});

fileInput.onchange = ({target})=>{
  let file = target.files[0];
  if(file){
    let fileName = file.name;
    // if(fileName.length >= 12){
    //   let splitName = fileName.split('.');
    //   fileName = splitName[0].substring(0, 13) + "... ." + splitName[1];
    // }
    uploadFile(fileName);
  }
}
var cancelButton = null;

function uploadFile(name){
  let xhr = new XMLHttpRequest(); //creating new XML object (AJAX)
  xhr.open("POST", "http://54.183.184.209/get_prediction_for_csv",true); //sending POST request to specifies URL/File
  console.log();
  debugger
  xhr.upload.addEventListener("progress", ({loaded, total}) =>{
    let fileLoaded = Math.floor((loaded / total) * 100); //getting percentage of loaded filesize
    let fileTotal = Math.floor(total / 1000); //getting filesize in KB from bytes
    let fileSize;
    (fileTotal < 1024) ? fileSize = fileTotal + " KB" : fileSize = (loaded / (1024*1024)).toFixed(2) + " MB";
    let progressHTML = `<li class="row">
                          <i class="fas fa-file-alt"></i>
                          <div class="content">
                            <div class="details">
                              <span class="name">${name} • Uploading</span>
                              <span class="percent">${fileLoaded}%</span>
                            </div>
                            <div class="progress-bar">
                              <div class="progress" style="width: ${fileLoaded}%"></div>
                            </div>
                          </div>
                        </li>`;
    uploadedArea.classList.add("onprogress");
    progressArea.innerHTML = progressHTML;
    if(loaded == total){
      progressArea.innerHTML = "";
      let uploadedHTML = `<li id="div-content" class="row">
                            <div class="content upload">
                              <i class="fas fa-file-alt"></i>
                              <div class="details">
                                <span class="name">${name} • Uploaded</span>
                                <span class="size">${fileSize}</span>
                                <button id="submitButton">Submit</button>
                              </div>
                            </div>
                            <i id="cancelButton" class="fa fa-window-close"></i>
                          </li>`;
                          uploadedArea.classList.remove("onprogress");
                          uploadedArea.insertAdjacentHTML("afterbegin", uploadedHTML);
        cancelButton = document.getElementById('cancelButton')
        cancelButton.addEventListener("click", function() {
          var deleteEntry = document.getElementById('div-content')
          deleteEntry.remove()
          xhr.abort()
        });
      // var button = document.getElementById("submitButton");
      // button.addEventListener("mouseenter", function() {
      //   // Add a CSS class to the button to apply the hover effect
      //   button.classList.add("elevate-on-hover");
      // });
      
      // // Add an event listener for mouseleave (hover out)
      // button.addEventListener("mouseleave", function() {
      //   // Remove the CSS class to reset the button's style
      //   button.classList.remove("elevate-on-hover");
      // });
    }
  });
  let formData = new FormData(form);
  xhr.send(formData);
}

// var cancelButton = document.getElementById('cancelButton')
// cancelButton.addEventListener("click", function() {
//   var deleteEntry = document.getElementById('div-content')
//   deleteEntry.remove()
// });