/*
 * @author Victor Holanda Rusu (CSCS)
 *
 */
var reframe =  {};
(function(namespace) {

var __cscsMarkDown = "";

/**
 *  @description reads file contents based on http request. it should be used by all markdown based pages
 *  @param {string} filename
 *  @param {function} callback
 *  @returns {void}
 */
namespace.read_file_contents = function (filename, callback)
{
  var rawFile = new XMLHttpRequest();
  rawFile.open("GET", filename, false);

  var __fileContent = "";
  rawFile.onreadystatechange = function ()
  {
    if(rawFile.readyState === 4) {
      if(rawFile.status === 200 || rawFile.status == 0) {
        __fileContent = rawFile.responseText;
        if(typeof callback == 'function'){
          callback(__fileContent);
        }
      }
    }
  }
  rawFile.send(null);
}

/**
 *  @description gets the base html and parses the sidebar using the same markdown used by remark.js
 *  before changing the markdown one needs to verify if it is compatible with the presentation mode
 *  @param {string} navbarfile
 *  @param {string} sidebarfile
 *  @returns {void}
 */
namespace.setup_site_content = function(navbarfile, sidebarfile) {

  namespace.read_file_contents(navbarfile, function __populate_site_content(argument) {
    document.getElementById("cscs-site-content").innerHTML = argument;
  });

  namespace.read_file_contents(sidebarfile, function cscs_populate_site_content(argument) {

    marked.setOptions({
      gfm: true,
      tables: true,
      breaks: false,
      pedantic: true,
      sanitize: false,
      smartLists: true,
      langPrefix: '',
    });
    marked(argument, function (err, content) {
      if (err) throw err;
      document.getElementById("cscs-leftbar-markdown").innerHTML = content;
    });

  });

  // presentation mode is hidden if remark.js is not included
  try {
    if(remark != null) {
      $('#start-cscs-presenter-mode').show();
    } else {
      $('#start-cscs-presenter-mode').hide();
    }
  } catch(msg) {
    $('#start-cscs-presenter-mode').hide();
  }

  var presenterMode = document.getElementById('start-cscs-presenter-mode');
  if (presenterMode != null) {
    presenterMode.onclick = namespace.__show_in_presenter_mode;    
  }

  namespace.__email_protector();
  namespace.__prepend_domain_to_links();
}

/**
 *  @description this is a wrapper to a selection of functions that need to be called after
 *  the main markdown is rendered
 */
namespace.__markdown_post_features = function() {
  namespace.__mouseover_link();
  namespace.__create_toc();
  namespace.__change_table_layout();
  namespace.__highlight_code();
}

/**
 *  @description parses the main website markdown using the same markdown used by remark.js
 *  before changing the markdown one needs to verify if it is compatible with the presentation mode
  * @param {string} navbarfile
  * @param {string} sidebarfile
  * @returns {void}
  */
namespace.setup_markdown_page_content = function(markdownFile) {
  __cscsMarkDown = markdownFile;

  namespace.read_file_contents(markdownFile, function __populate_site_content(argument) {

    marked.setOptions({
      gfm: true,
      tables: true,
      breaks: false,
      pedantic: true,
      sanitize: false,
      smartLists: true,
      langPrefix: '',
    });
    marked(argument, function (err, content) {
      if (err) throw err;
      document.getElementById("cscs-markdown-content").innerHTML = content;
    });
  });
  namespace.__markdown_post_features();
}

/**
 *  @description reads the news markdown and appends it to the cscs-markdown-content.
 *  number_of_news controls the number of news items to print. If negative, all news are printed.
 *  @param {string} news_markdown_file
 *  @param {number} number_of_news
 *  @returns {void}
 */
namespace.read_news = function(news_markdown_file, number_of_news)
{
  namespace.read_file_contents(news_markdown_file, function __populate_site_content(argument) {
    marked(argument, function (err, content) {
      if (err) throw err;

      $('#toc').children().remove();

      var theStart = namespace.__getCommentsObject('#cscs-markdown-content', ' start-news ');
      $(content).insertAfter(theStart);
      var theEnd = namespace.__getCommentsObject('#cscs-markdown-content', ' end-news ');

      if(Number(number_of_news) > 0) {
        var lastOne = $(theEnd).prev();
        if($(theStart).nextUntil(lastOne).filter("h2").length > number_of_news) {
          toRemove = $(theStart).nextUntil(lastOne).filter("h2").slice(0, number_of_news + 1).last();
          toRemove.nextUntil(lastOne).remove();
          toRemove.remove();
          lastOne.remove();
        }
      }
      namespace.__markdown_post_features();
    });
  });
}

/**
 *  @description hides the right toc from the page and makes it a two column page.
 *  @param {string} markdownFile
 *  @returns {void}
 */
namespace.two_column_mode = function(markdownFile) {

  var markdown_div = $('#cscs-markdown-content');
  if (markdown_div.hasClass('col-md-7') == true) {
    markdown_div.removeClass('col-md-7');
  }
  markdown_div.addClass('col-md-9');

  $('#cscs-rightbar').hide();

}

// function cscs_get_modulelist(link, regex, shash_at = 0, elementid = "cscs-markdown-content")
// {
//   read_file_contents(link, function __populate_site_content(argument) {
//     var pattern = regex;
//     var parsed_module = "";

//     var result = pattern.exec(argument);
//     while (result) {
//       var holder = result + '';
//       holder = holder.replace('.eb', '');
//       var splitter = holder.split('-');
//       holder = "";
//       cat = '-';
//       for (var i = 0; i < splitter.length; i++) {
//         if(i == shash_at) {
//           cat = '/';
//         } else if(i == splitter.length -1){
//           cat = '';
//         } else {
//           cat = '-';
//         }
//         holder += splitter[i] + cat;
//       }
//       parsed_module += holder + '\n';
//       result = pattern.exec(argument);
//     }
//     document.getElementById(elementid).innerHTML += parsed_module;
//   });
// }

/**
 *  @description starts the presentation mode. this mode destroys completely the page.
 *  So we refresh the page if the exit button is clicked.
 *  @returns {void}
 */
namespace.__show_in_presenter_mode= function() {
  document.getElementById("cscs-body-container").innerHTML = null;

  try {
    var slideshow = remark.create({sourceUrl: __cscsMarkDown});

    // the click blocks, so forcing full page reload for the click
    var presenterMode = document.getElementById('start-cscs-presenter-mode');
    presenterMode.style.display = 'none';
    var exitMode = document.getElementById('exit-cscs-presenter-mode');
    exitMode.style.display = 'block';

    // workaround to work for local and non local servers
    if(document.location.domain == null)
      exitMode.href = document.location.pathname;
    else
      exitMode.href = document.location.domain+document.location.pathname;

    slideshow.on();
  } catch(msg) {
    $('#start-cscs-presenter-mode').hide();
  }
}

/**
 *  @description protects cscs' e-mail from robots
 *  @returns {void}
 */
namespace.__email_protector = function() {
  // $("#cscs-email-protector").prepend('<a href="&#109;&#97;&#105;&#108;&#116;&#111;&#58;%69%6E%66%6F%40%63%73%63%73%2E%63%68">Contact CSCS</a>');
  $("#cscs-email-protector").prepend('<a href="&#109;&#097;&#105;&#108;&#116;&#111;:&#104;&#101;&#108;&#112;&#064;&#099;&#115;&#099;&#115;&#046;&#099;&#104;">Contact us</a>');
}

/**
 *  @description wraps the markdown headers with a link it does it on mouse hover.
 *  But it is not really needed. We could do it immediately without the need of hovering
 *  @returns {void}
 */
namespace.__mouseover_link = function() {

  $('#cscs-markdown-content').children("h1, h2, h3").each(function(index, element) {
    $(element).hover(
      function() {
        $(this).wrap(function() {
          return "<a id='" + $(this).attr('id') + "' href='#" + $( this ).attr('id') + "'></a>";
        });
      },
      function() {
        $(this).unwrap();
      }
    );
  });
}

/**
 *  @description creates the toc based on the main markdown content.
 *  @returns {void}
 */
namespace.__create_toc = function() {
  // TOC creation
  // $('#toc').TOC();
  $('#toc').TOC("#cscs-markdown-content", {
    headings: ["h1","h2"],
  });

  $('#toc').append('<a class="back-to-top" href="#">Back to top</a>');
  // Sidenav affixing
  setTimeout(function () {
    $('#toc').affix({
      offset: {
        // top: $('.cscs-global-nav').outerHeight(),
        top: function () {
          return $('.cscs-global-nav').outerHeight()
        },
        bottom: function () {
          return (this.bottom = $('.footer').outerHeight(true));
        }

      }
    })}, 100);
  var $window = $(window);
  var $body   = $(document.body);

  $body.scrollspy({
    target: '#toc'
  });

  $window.on('load', function () {
    $body.scrollspy('refresh');
  });

  // counting the number of h1 levels in the toc
  var tocsize = 0;
  $('#toc').children("ul").each(function(index, element) {
    var size = $(this).children("li").length;
    if(size > tocsize){
      tocsize = size;
    }
  });

  // expanding toc when there is only one h1
  if(tocsize == 1) {
    $("head").append("<style> nav[data-toggle='toc'] .nav .nav { display: block; } </style>");
  }
}

/**
 *  @description destroys the remark presentation and restores the CSCS website. This should be deprecated
 *  @returns {void}
 */
namespace.__exit_presentation_mode = function() {
  if(document.location.domain == null)
    window.location.assign(document.location.pathname);
  else
    window.location.assign(document.location = document.location.domain+document.location.pathname);
}

/**
 *  @description changes the layout of table inside the main side markdown area.
 *  @returns {void}
 */
namespace.__change_table_layout = function() {
  $('#cscs-markdown-content').children("table").each(function(index, element) {
    $(element).addClass('table table-striped table-bordered' );
    $(this).wrap(function() {
      return "<div class='table-responsive' ></a>";
    });
  });
}

/**
 *  @description highlights `pre code` blocks.
 *  @returns {void}
 */
namespace.__highlight_code = function() {
  try {
      $('pre').addClass('line-numbers');

      $('pre code').addClass(function( index, currentClass ) {
        var addedClass = "language-" + currentClass;
        return addedClass;
      });
  } catch(msg) {
      console.log('error');
      console.log(msg);
    // no need to catch
  }
}

/**
 *  @description prepends the domain to the navbar.
 *  @returns {void}
 */
namespace.__prepend_domain_to_links = function()
{
  // var domain = window.location.origin;
  var domain = window.location.host + "/";
  // var domain = "";

  if(document.location.domain != null) {
    domain = document.location.domain;
  }
  domain += "/";

  // console.log('window.location.origin: ' + window.location.origin);
  // console.log('window.location.host: ' + window.location.host);
  // console.log('window.location.hostname: ' + window.location.hostname);
  // console.log('window.location.pathname: ' + window.location.pathname);
  // console.log('window.location.domain: ' + window.location.domain);

  var folders = [ 'about', 'pipeline', 'running', 'started', 'structure', 'usecases', 'writing_checks', '' ]
  var paths = window.location.pathname.split('/');
  paths.forEach(function(element) {
    // console.log(element);
    if ((folders.indexOf(element) == -1) && (element.indexOf('.') == -1)) {
      domain += element + "/";
    }
  }, this);

  domain = domain.replace('//', '/');
  domain = domain.replace('//', '/');
  domain = domain.replace('//', '/');
//   var domain=domain.split('/').filter(function(item,i,allItems){
//     return i==allItems.indexOf(item);
// }).join('/');
//   console.log('domain: ' + domain);

  // prepending cscs domain
  $('.reframe-prepend-domain').each(function(index, element) {
    $(element).attr('href',function(i,v) {
      return window.location.protocol + '//' + (domain + v).replace('//', '/');
    });
  });
  
  $('#cscs-leftbar-markdown').children('h1').children('a').each(function(index, element) {
    $(element).attr('href',function(i, v) {
      return window.location.protocol + '//' + (domain + v).replace('//', '/');
    });
  });  

}

/**
 *  @description helper function to get the position of the comments in the page.
 *  This is important for the news pages.
 *  @returns {void}
 */
namespace.__getCommentsObject = function(element, comment) {
  var returnValue = null;
  $(element).contents().filter(function(){
    return this.nodeType == 8;
  }).each(function(i, e){
    if(e.nodeValue === comment) {
      returnValue = e;
      return false;
    }
  });
  return returnValue;
}
})(reframe);
