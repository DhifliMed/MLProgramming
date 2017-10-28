
function delmodel() {
            myModal.classList.remove('show');
            myModal.classList.add('fade');

        }
function changealg() {
          //  var x=document.getElementById('id0');
            var e = document.getElementById("id0");
            var value = e.options[e.selectedIndex].value;

            document.getElementById('id1').classList.remove('visible');
            document.getElementById('id2').classList.remove('visible');
            document.getElementById('id3').classList.remove('visible');
            document.getElementById('id1').classList.add('hidden');
            document.getElementById('id2').classList.add('hidden');
            document.getElementById('id3').classList.add('hidden');

            if(value=='0') {
                document.getElementById('id1').classList.remove('hidden');
                document.getElementById('id1').classList.add('visible show');

            }
            else if(value=='1') {
                document.getElementById('id2').classList.remove('hidden');
                document.getElementById('id2').classList.add('visible');
            }
            else if(value=='2') {
                document.getElementById('id3').classList.remove('hidden');
                document.getElementById('id3').classList.add('visible');
            }
            else{
                alert('error');
            }


        }

  $( document ).ready(function(){
//   Hide the border by commenting out the variable below
    var $on = 'cb';
    $($on).css({
      'background':'none',
      'border':'none',
      'box-shadow':'none'
    });
});