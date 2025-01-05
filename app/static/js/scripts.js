document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('prediction-form');
    const fileInput = document.getElementById('file-upload');
    const fileLabel = document.querySelector('label[for="file-upload"]');
    const results = document.getElementById('results');
    const loading = document.getElementById('loading');
    const errorModal = document.getElementById('error-modal');
    const errorMessage = document.getElementById('error-message');
    const closeError = document.getElementById('close-error');

    console.log('DOM fully loaded and parsed');

    // Update file label when a file is selected
    fileInput.addEventListener('change', function () {
        console.log('File input changed');
        if (fileInput.files.length > 0) {
            fileLabel.textContent = fileInput.files[0].name;
            console.log('File selected:', fileInput.files[0].name);
        } else {
            fileLabel.textContent = 'Choose a file';
            console.log('No file selected');
        }
    });

    // Handle form submission
    form.addEventListener('submit', function (e) {
        e.preventDefault();
        console.log('Form submitted');
        loading.classList.remove('hidden');
        results.classList.add('hidden');

        const formData = new FormData(form);
        
        // Log form data
        for (let [key, value] of formData.entries()) {
            console.log(key, value);
        }

        fetch('/upload', {
            method: 'POST',
            body: formData,
        })
            .then((response) => {
                console.log('Response status:', response.status);
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then((data) => {
                console.log('Response data:', data);
                loading.classList.add('hidden');
                if (data.error) {
                    showError(data.error);
                } else {
                    displayResults(data);
                }
            })
            .catch((error) => {
                console.error('Fetch error:', error);
                loading.classList.add('hidden');
                showError('An error occurred. Please try again.');
            });
    });

    function displayResults(data) {
        results.classList.remove('hidden');
        var table = document.getElementById('prediction-table');
        var tableHTML = '<table class="w-full border-collapse border border-gray-300">' +
            '<thead><tr class="bg-gray-100">' +
            '<th class="border border-gray-300 px-4 py-2">Date</th>' +
            '<th class="border border-gray-300 px-4 py-2">Predicted Max Temperature (°C)</th>' +
            '</tr></thead><tbody>';

        data.predictions.forEach(function(prediction) {
            tableHTML += '<tr>' +
                '<td class="border border-gray-300 px-4 py-2">' + prediction.date + '</td>' +
                '<td class="border border-gray-300 px-4 py-2">' + prediction.predicted_tempmax.toFixed(2) + '</td>' +
                '</tr>';
        });

        tableHTML += '</tbody></table>';
        table.innerHTML = tableHTML;

        document.getElementById('temp-over-time').src = '/static/outputs/temperature_over_time.png';
        document.getElementById('temp-distribution').src = '/static/outputs/temperature_distribution.png';
        document.getElementById('correlation-heatmap').src = '/static/outputs/correlation_heatmap.png';
        document.getElementById('download-csv').href = '/download/' + data.csv_filename;

        results.scrollIntoView({ behavior: 'smooth' });
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorModal.classList.remove('hidden');
    }

    if (closeError) {
        closeError.addEventListener('click', function() {
            errorModal.classList.add('hidden');
        });
    }

    var contactForm = document.getElementById('contact-form');
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            alert('Thank you for your message. We will get back to you soon!');
            contactForm.reset();
        });
    }
});