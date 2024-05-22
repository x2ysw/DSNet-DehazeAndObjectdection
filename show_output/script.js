document.addEventListener("DOMContentLoaded", function() {
    const imageUpload = document.getElementById("imageUpload");
    const uploadButton = document.getElementById("uploadButton");
    const originalImage = document.getElementById("originalImage");
    const dehazedImage = document.getElementById("dehazedImage");
    const detectionImage = document.getElementById("detectionImage");
    const progressBar = document.querySelector(".progress");

    uploadButton.addEventListener("click", function() {
        if (imageUpload.files.length === 0) {
            alert("Please select an image to upload.");
            return;
        }

        const file = imageUpload.files[0];
        const formData = new FormData();
        formData.append("image", file);

        // Display original image
        const reader = new FileReader();
        reader.onload = function(e) {
            originalImage.src = e.target.result;
        };
        reader.readAsDataURL(file);

        // Simulate progress
        let progress = 0;
        const interval = setInterval(() => {
            if (progress < 100) {
                progress += 1;
                progressBar.style.width = progress + "%";
            } else {
                clearInterval(interval);
            }
        }, 50);

        // Send image to the backend API
        fetch("your_backend_api_url", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Update progress to 100% after receiving response
            progressBar.style.width = "100%";

            // Display dehazed and detection images
            dehazedImage.src = data.dehazedImageUrl;
            detectionImage.src = data.detectionImageUrl;
        })
        .catch(error => {
            console.error("Error:", error);
            alert("An error occurred while processing the image.");
        });
    });
});
