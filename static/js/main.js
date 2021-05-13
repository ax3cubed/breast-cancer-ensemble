$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });
 
    // Predict
    $('#btn-predict').click(function () {
        $('.loader').hide();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        var form_data = new FormData($('#upload-file')[0]);
        form_data.append('ensemble', $('select#ensemble').val())
        // Show loading animation
        $(this).hide();
        $('.loader').show();
        form_data.forEach( x =>console.log(x))
        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(' Model Predict the sample as:  ' + data);
                console.log('Success!');
            },
            error: function (data) {
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text('an error just occured', data);
                console.log(data, form_data);
            }
        });
    });

});
