
$( document ).ready(function() {

    $("#translate_btn").click(function(e){
        
        $.ajax({
            type: "GET",
            url: "/translate",
            data: {
                query: $("#input_field").val(),
            },
            success: function(response){
                $("#output_field").text(response.result)
            }
        })

    });

});

