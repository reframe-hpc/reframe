// https://github.com/MadeByMike/jQuery-table-of-contents

(function($){
$.TOC = function(el, scope, options){
    var toc = this;
    toc.$el = $(el);
    toc.html = "";

    toc.init = function(){
        $.extend(options, toc.$el.data());
        for (var key in options) {
            if (options.hasOwnProperty(key)) {
                if(options[key] == ""){
                    delete options[key];
                }
            }
        }
        toc.options = $.extend({},$.TOC.defaultOptions, options);

        if(typeof(toc.options.scope) == "undefined" || toc.options.scope == null){ toc.options.scope = document.body; }
        if(typeof(options) == "undefined" || options == null){ options = ''; }

        // toc.options.nested = (options.headings) ? false : toc.options.nested; // Can only be nested if headings are default h1-h6
        toc.options.nested = true;
        toc.options.startLevel = (toc.options.startLevel < 1 || toc.options.startLevel > 6) ?  1 : toc.options.startLevel; // Validate starting level
        toc.options.depth =  (toc.options.depth < 1 || toc.options.depth > 6) ? 1 : toc.options.depth; // Validate depth
        var filtered_tags = (toc.options.nested) ? toc.options.headings.splice(toc.options.startLevel - 1, toc.options.depth) : toc.options.headings; // If nested get only the tags starting with startLevel, and counting to depth

        // Cache all the headings and strip those to be ignored
        scope = toc.options.scope;
        toc.$headings = $(scope).children(filtered_tags.join(', ')).filter(function(){
            if ($(this).closest(toc.options.ignoreContainers).length === 0){
                return true;
            }
            return false;
        });
        //Exit if no headings
        if(toc.$headings.length == 0){ return; }

        // If topLinks is enabled, set/get an id for the body element
        if(toc.options.topLinks !== false){
            var id = $(document.body).attr('id');
            if(id == "") {
                id = toc.options.topBodyId;
                document.body.id = id;
            }
            // Cache the id locally
            toc.topLinkId = id;
        }
        toc.$el.append("<"+ toc.options.containerType +" class='nav'>"+toc.buildTOC()+"</"+ toc.options.containerType +">");
        return toc; // Return this object for memory cleanup
    };

    toc.buildTOC = function(){
        toc.current_depth = toc.options.startLevel;
        toc.$headings.each(function(i){
            var depthClass = this.nodeName.toLowerCase();
            if(toc.options.nested){
                // Get current depth based on h1, h2, h3, etc.
                var depth = depthClass.substr(1,1);
                // This changes depth, or adds separators, only if not the first item
                if(i > 0 || ( i == 0 && depth != toc.current_depth)){
                    toc.addItem(depth, i);
                }
                toc.html += toc.formatLink(this, depth, i) + "\n";
            } else {
                toc.html += "<"+toc.options.itemType+">\n" + toc.formatLink(this, depthClass, i) + "\n" + "</"+toc.options.itemType+">\n";
            }
            if(toc.options.topLinks !== false) { toc.addTopLink(this); }

        });
        toc.addItem(toc.options.startLevel, toc.$headings.length);

        if(toc.options.nested) { toc.html = "<"+toc.options.itemType+"  class=''>\n" + toc.html + "</"+toc.options.itemType+">\n"; }

        return toc.html;
    };

    toc.addTopLink = function(element){
        var text = (toc.options.topLinks === true ? "Top" : toc.options.topLinks );
        var $a = $("<a href='#" + toc.topLinkId + "' class='" + toc.options.topLinkClass + "'></a>").html(text);
        $(element).append($a);
    };

    toc.formatLink = function(element, depthClass, index){
        // Get a unique id for the element
        var id = element.id;
        if(id == ""){
            id = "toc-id-"+index;
            element.id = id;
        }
        return '<a href="#' + id + '" class="' + toc.formatDepthClass(depthClass) + '" >' + toc.options.linkText.replace('%', $(element).text()) + '</a>';
    };

    toc.addItem = function(new_depth, index){
        var i;
        if(new_depth > toc.current_depth){
            // Opening tags for changes in depth
            for(i = toc.current_depth; i < new_depth; i++){
                toc.html += '<' + toc.options.containerType + ' class="nav">' + "\n <"+toc.options.itemType+">\n";
            }
        } else if (new_depth < toc.current_depth){
            // Account for changes in depth
            for(i = toc.current_depth; i > new_depth; i--){
                toc.html += "</" + toc.options.itemType+">\n </" + toc.options.containerType + '>' + "\n";
            }
            // Open next block
            if ((toc.$headings.length) != index) {
                toc.html += "</"+toc.options.itemType+">\n<"+toc.options.itemType+">\n";
            }
        } else {
            // Just close a tag and open a new one since the depth has not changed
            if ((toc.$headings.length) != index) {
                toc.html += "</"+toc.options.itemType+">\n<"+toc.options.itemType+">\n";
            }
        }
        toc.current_depth = new_depth;
    };

    toc.formatDepthClass = function(depthClass){
        if(toc.options.nested){
            // Normalizes the depth to always start at 1, even if the starting tier is > 1
            return toc.options.depthClass.replace('%', (depthClass - ( toc.options.startLevel - 1 ) ) );
        } else {
            // Otherwise just nodename
            return toc.options.depthClass.replace('%', depthClass );
        }
    };

    return toc.init(scope,options);
};

$.TOC.defaultOptions = {
    scope: "#cscs-markdown-content",
    headings: ["h1","h2"],
    startLevel: 1,
    depth: 2,
    depthClass: "toc-depth-%",
    linkText: "%",
    topLinks: false,
    topLinkClass: "toc-link",
    topBodyId: "top",
    containerType: 'ul',
    itemType: 'li',
    nested: true,
    ignoreContainers: '' // the script will ignore these containers
};


$.fn.TOC = function(scope, options){
    return this.each(function(){
        var toc = new $.TOC(this, scope, options);
        delete toc;
    });
};
})(jQuery);