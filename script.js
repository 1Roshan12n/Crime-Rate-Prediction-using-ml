// static/js/script.js
$(document).ready(function() {
    // Handle form submission
    $('#predictionForm').on('submit', function(e) {
        e.preventDefault();
        
        // Show loading indicator
        $('#no-results').html('<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div>');
        
        // Get form data
        const formData = new FormData(this);
        
        // Send AJAX request
        $.ajax({
            url: '/predict',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                // Display prediction
                $('#prediction-value').text(response.prediction);
                
                // Calculate total importance for normalization
                const importanceValues = Object.values(response.feature_importance);
                const maxImportance = Math.max(...importanceValues);
                
                // Update feature importance bars
                for (const [feature, importance] of Object.entries(response.feature_importance)) {
                    const percentImportance = (importance / maxImportance * 100).toFixed(1);
                    $(`#importance-${feature}`)
                        .css('width', `${percentImportance}%`)
                        .text(`${(importance * 100).toFixed(1)}%`);
                }
                
                // Show results
                $('#results').removeClass('d-none');
                $('#no-results').addClass('d-none');
            },
            error: function(error) {
                console.error('Error:', error);
                $('#no-results').html('<div class="alert alert-danger">Error making prediction. Please try again.</div>');
            }
        });
    });
    
    // Handle model training
    $('#trainModel').on('click', function() {
        // Show loading indicator
        $(this).prop('disabled', true).html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Training...');
        $('#trainingResults').removeClass('d-none').html('Training model, please wait...');
        
        // Send AJAX request
        $.ajax({
            url: '/train',
            type: 'POST',
            success: function(response) {
                // Update button
                $('#trainModel').prop('disabled', false).text('Train/Retrain Model');
                
                // Display training results
                let resultsHtml = `
                    <h4>Model Training Complete</h4>
                    <p><strong>Training Score:</strong> ${(response.train_score * 100).toFixed(2)}%</p>
                    <p><strong>Testing Score:</strong> ${(response.test_score * 100).toFixed(2)}%</p>
                    <h5>Feature Importance:</h5>
                    <ul>
                `;
                
                // Sort features by importance
                const sortedFeatures = Object.entries(response.feature_importance)
                    .sort((a, b) => b[1] - a[1]);
                
                for (const [feature, importance] of sortedFeatures) {
                    resultsHtml += `<li>${feature}: ${(importance * 100).toFixed(2)}%</li>`;
                }
                
                resultsHtml += '</ul>';
                
                $('#trainingResults').html(resultsHtml);
            },
            error: function(error) {
                console.error('Error:', error);
                $('#trainModel').prop('disabled', false).text('Train/Retrain Model');
                $('#trainingResults').html('<div class="alert alert-danger">Error training model. Please try again.</div>');
            }
        });
    });
});