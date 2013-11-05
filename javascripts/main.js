console.log('This would be the main JS file.');

// JavaScript functions to show and hide drop-down menus.
// In SimpleNavBar.html we call ShowMenuDiv each time the mouse goes over 
// either the menu title or the submenu itself, and call HideMenuDiv when the
// mouse goes out of the menu title or the submenu iteslf (onMouseOut).

function ShowItem (itemID) {
  var x = document.getElementById(itemID);
  if (x)
    x.style.visibility = "visible";
  return true;
}

function HideItem (itemID) { 
  var x = document.getElementById(itemID);
  if (x)
     x.style.visibility = "hidden";
  return true;
}

//    As noted in the css file, using x.style.visibility as
//    seen below seemed to have better cross browser support than using 
//    x.style.display="block" and x.style.display="none" to show and hide 
//    the menu.
