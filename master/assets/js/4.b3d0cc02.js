(window.webpackJsonp=window.webpackJsonp||[]).push([[4],{323:function(t,e,a){"use strict";a(359),a(100),a(72);var n=a(318),i={name:"NavLink",props:{item:{required:!0}},computed:{link:function(){return Object(n.b)(this.item.link)},exact:function(){var t=this;return this.$site.locales?Object.keys(this.$site.locales).some((function(e){return e===t.link})):"/"===this.link},isNonHttpURI:function(){return Object(n.g)(this.link)||Object(n.h)(this.link)},isBlankTarget:function(){return"_blank"===this.target},isInternal:function(){return!Object(n.f)(this.link)&&!this.isBlankTarget},target:function(){return this.isNonHttpURI?null:this.item.target?this.item.target:Object(n.f)(this.link)?"_blank":""},rel:function(){return this.isNonHttpURI?null:this.item.rel?this.item.rel:this.isBlankTarget?"noopener noreferrer":""}},methods:{focusoutAction:function(){this.$emit("focusout")}}},s=a(26),r=Object(s.a)(i,(function(){var t=this,e=t.$createElement,a=t._self._c||e;return t.isInternal?a("RouterLink",{staticClass:"nav-link",attrs:{to:t.link,exact:t.exact},nativeOn:{focusout:function(e){return t.focusoutAction(e)}}},[t._v("\n  "+t._s(t.item.text)+"\n")]):a("a",{staticClass:"nav-link external",attrs:{href:t.link,target:t.target,rel:t.rel},on:{focusout:t.focusoutAction}},[t._v("\n  "+t._s(t.item.text)+"\n  "),t.isBlankTarget?a("span",{staticClass:"external__icon"},[a("vp-icon",{attrs:{color:"#4A4A4A",name:"blank",size:"12px"}})],1):t._e()])}),[],!1,null,null,null);e.a=r.exports},325:function(t,e,a){},326:function(t,e,a){},334:function(t,e,a){},337:function(t,e,a){},349:function(t,e,a){},350:function(t,e,a){},351:function(t,e,a){},362:function(t,e,a){"use strict";a(325)},365:function(t,e,a){"use strict";a(326)},373:function(t,e,a){"use strict";a(334)},376:function(t,e,a){"use strict";a(337)},399:function(t,e,a){"use strict";a(349)},400:function(t,e,a){"use strict";a(350)},401:function(t,e,a){"use strict";a(351)},408:function(t,e,a){"use strict";var n=a(318),i=a(375),s=a.n(i),r=a(335),o=a.n(r),l={name:"PageNav",props:["sidebarItems"],computed:{prev:function(){return u(c.PREV,this)},next:function(){return u(c.NEXT,this)}}};var c={NEXT:{resolveLink:function(t,e){return h(t,e,1)},getThemeLinkConfig:function(t){return t.nextLinks},getPageLinkConfig:function(t){return t.frontmatter.next}},PREV:{resolveLink:function(t,e){return h(t,e,-1)},getThemeLinkConfig:function(t){return t.prevLinks},getPageLinkConfig:function(t){return t.frontmatter.prev}}};function u(t,e){var a=e.$themeConfig,i=e.$page,r=e.$route,l=e.$site,c=e.sidebarItems,u=t.resolveLink,h=t.getThemeLinkConfig,p=t.getPageLinkConfig,d=h(a),v=p(i),g=o()(v)?d:v;return!1===g?void 0:s()(g)?Object(n.k)(l.pages,g,r.path):u(i,c)}function h(t,e,a){var n=[];!function t(e,a){for(var n=0,i=e.length;n<i;n++)"group"===e[n].type?t(e[n].children||[],a):a.push(e[n])}(e,n);for(var i=0;i<n.length;i++){var s=n[i];if("page"===s.type&&s.path===decodeURIComponent(t.path))return n[i+a]}}var p=l,d=(a(376),a(26)),v=Object(d.a)(p,(function(){var t=this,e=t.$createElement,a=t._self._c||e;return t.prev||t.next?a("div",{staticClass:"page-nav"},[a("p",{staticClass:"inner"},[t.prev?a("span",{staticClass:"page-nav__button prev"},["external"===t.prev.type?a("a",{staticClass:"prev",attrs:{href:t.prev.path,target:"_blank",rel:"noopener noreferrer"}},[a("span",{staticClass:"page-nav__button__icon"},[a("vp-icon",{attrs:{color:"#4A4A4A",name:"chev-left",size:"18px"}})],1),t._v(" "),a("span",{staticClass:"page-nav__button__text"},[t._v("\n          "+t._s(t.prev.title||t.prev.path)+"\n        ")]),t._v(" "),a("OutboundLink")],1):a("RouterLink",{staticClass:"prev",attrs:{to:t.prev.path}},[a("span",{staticClass:"page-nav__button__icon"},[a("vp-icon",{attrs:{color:"#4A4A4A",name:"chev-left",size:"18px"}})],1),t._v(" "),a("span",{staticClass:"page-nav__button__text"},[t._v("\n          "+t._s(t.prev.title||t.prev.path)+"\n        ")])])],1):t._e(),t._v(" "),t.next?a("span",{staticClass:"page-nav__button next"},["external"===t.next.type?a("a",{attrs:{href:t.next.path,target:"_blank",rel:"noopener noreferrer"}},[a("span",{staticClass:"page-nav__button__text"},[t._v("\n          "+t._s(t.next.title||t.next.path)+"\n        ")]),t._v(" "),a("span",{staticClass:"page-nav__button__icon"},[a("vp-icon",{attrs:{color:"#4A4A4A",name:"chev-right",size:"18px"}})],1),t._v(" "),a("OutboundLink")],1):a("RouterLink",{attrs:{to:t.next.path}},[a("span",{staticClass:"page-nav__button__text"},[t._v("\n          "+t._s(t.next.title||t.next.path)+"\n        ")]),t._v(" "),a("span",{staticClass:"page-nav__button__icon"},[a("vp-icon",{attrs:{color:"#4A4A4A",name:"chev-right",size:"18px"}})],1)])],1):t._e()])]):t._e()}),[],!1,null,null,null);e.a=v.exports},412:function(t,e,a){"use strict";a.r(e);var n={name:"Home",components:{NavLink:a(323).a},computed:{data:function(){return this.$page.frontmatter},actionLink:function(){return{link:this.data.actionLink,text:this.data.actionText}}},mounted:function(){document.querySelector(".global-ui").classList.add("hidden")}},i=(a(362),a(26)),s=Object(i.a)(n,(function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",[a("div",{staticClass:"home__nav"},[a("RouterLink",{attrs:{to:"/"}},[t.data.navImage?a("img",{staticClass:"home__nav__logo",attrs:{src:t.$withBase(t.data.navImage)}}):t._e()])],1),t._v(" "),a("main",{staticClass:"home",attrs:{"aria-labelledby":"main-title"}},[a("header",{staticClass:"hero"},[t.data.heroImage?a("img",{attrs:{src:t.$withBase(t.data.heroImage),alt:t.data.heroAlt||"hero"}}):t._e(),t._v(" "),null!==t.data.heroText?a("h1",{attrs:{id:"main-title"}},[t._v("\n        "+t._s(t.data.heroText||t.$title||"Hello")),a("span",[t._v(t._s(t.data.heroSubText))])]):t._e(),t._v(" "),null!==t.data.tagline?a("p",{staticClass:"description"},[t._v("\n        "+t._s(t.data.tagline||t.$description||"Welcome to your VuePress site")+"\n      ")]):t._e(),t._v(" "),t.data.actionText&&t.data.actionLink?a("p",{staticClass:"action"},[a("NavLink",{staticClass:"action-button",attrs:{item:t.actionLink}})],1):t._e()]),t._v(" "),t.data.features&&t.data.features.length?a("div",{staticClass:"features"},t._l(t.data.features,(function(e,n){return a("div",{key:n,staticClass:"feature"},[a("h2",[t._v(t._s(e.title))]),t._v(" "),a("p",[t._v(t._s(e.details))]),t._v(" "),a("span",{staticClass:"feature__images"},[e.img1?a("img",{attrs:{src:t.$withBase(e.img1)}}):t._e(),t._v(" "),e.img2?a("img",{attrs:{src:t.$withBase(e.img2)}}):t._e(),t._v(" "),e.img3?a("img",{attrs:{src:t.$withBase(e.img3)}}):t._e()])])})),0):t._e(),t._v(" "),a("Content",{staticClass:"theme-default-content custom"}),t._v(" "),a("div",{staticClass:"footer"},[a("div",[t._v("\n        "+t._s(t.data.footer)+"\n        "),a("a",{attrs:{href:"https://recogn.ai",target:"_blank"}},[a("img",{attrs:{width:"70px",src:t.$withBase("/assets/img/recognai.png")}})])])])],1),t._v(" "),a("svg",{staticClass:"home__bg",attrs:{width:"494px",height:"487px",viewBox:"0 0 494 487",version:"1.1",xmlns:"http://www.w3.org/2000/svg","xmlns:xlink":"http://www.w3.org/1999/xlink"}},[a("g",{attrs:{id:"*-Documentation",stroke:"none","stroke-width":"1",fill:"none","fill-rule":"evenodd"}},[a("g",{attrs:{id:"bg",transform:"translate(-967.000000, 0.000000)",stroke:"#4C1DBF"}},[a("g",{attrs:{id:"Page-1",transform:"translate(1283.285018, 114.169241) scale(1, -1) rotate(-222.000000) translate(-1283.285018, -114.169241) translate(905.285018, -340.330759)"}},[a("path",{attrs:{d:"M190.848191,92.6665641 C190.848191,192.936685 213.213577,625.695092 235.586289,736.523335 C254.765514,831.56423 285.912986,889.573385 330.651084,889.573385 C364.200995,889.573385 384.068724,834.404885 403.343168,720.689711 C425.715879,588.752345 436.900404,182.385682 436.900404,97.9420653 C436.900404,48.4228362 434.966734,0 317.536552,0 C200.099045,0 190.848191,42.0900989 190.848191,92.6665641 Z",id:"Stroke-1"}}),t._v(" "),a("path",{attrs:{d:"M462.669807,123.072945 C430.876905,211.06309 314.908331,598.02194 301.004923,702.481229 C289.087189,792.060095 300.264439,852.999143 342.737252,867.408306 C374.592783,878.218748 410.946475,836.208684 465.303866,742.63481 C528.370228,634.062698 667.827973,281.070342 694.60327,206.971687 C710.30447,163.519279 723.824742,120.402466 612.328543,82.569488 C500.836027,44.7365105 478.706251,78.6922951 462.669807,123.072945 Z",id:"Stroke-3"}}),t._v(" "),a("path",{attrs:{d:"M522.165599,265.889246 C484.284461,338.587617 340.934261,662.14445 319.206142,752.302566 C300.572019,829.62022 306.700936,885.333623 346.978259,904.95665 C377.187174,919.671243 415.917499,888.385777 476.23195,814.402289 C546.21606,728.556455 709.802763,438.844613 741.702669,377.626624 C760.410634,341.725445 776.966094,305.770719 671.234891,254.266074 C565.503688,202.761429 541.272313,229.224136 522.165599,265.889246 Z",id:"Stroke-5"}}),t._v(" "),a("path",{attrs:{d:"M7.4784994,296.640189 C36.7838113,378.827884 184.591094,726.457068 238.303966,810.212619 C284.362025,882.038214 330.995856,919.720195 373.625119,905.548919 C405.598912,894.923135 408.403955,843.411958 393.53723,744.099005 C376.28991,628.87232 268.166596,292.24663 243.482222,223.032939 C229.006727,182.4433 213.014294,143.364503 101.111555,180.55831 C-10.791185,217.752117 -7.30702698,255.184666 7.4784994,296.640189 Z",id:"Stroke-7"}}),t._v(" "),a("path",{attrs:{d:"M222.393346,293.866117 C222.393346,371.203174 244.758733,704.97758 267.127782,790.456697 C286.31067,863.760269 317.458142,908.500479 362.196239,908.500479 C395.746151,908.500479 415.61388,865.948649 434.888324,778.245395 C457.261035,676.485733 468.445559,363.064689 468.445559,297.935359 C468.445559,259.74241 466.511889,222.393346 349.078045,222.393346 C231.644201,222.393346 222.393346,254.857889 222.393346,293.866117 Z",id:"Stroke-9"}})])])])])])}),[],!1,null,null,null).exports,r=(a(363),a(99)),o=(a(9),a(10),a(16),a(179),a(47),a(319),{name:"AlgoliaSearchBox",props:["options"],data:function(){return{placeholder:void 0}},watch:{$lang:function(t){this.update(this.options,t)},options:function(t){this.update(t,this.$lang)}},mounted:function(){this.initialize(this.options,this.$lang),this.placeholder=this.$site.themeConfig.searchPlaceholder||""},methods:{initialize:function(t,e){var n=this;Promise.all([Promise.all([a.e(0),a.e(2)]).then(a.t.bind(null,410,7)),Promise.all([a.e(0),a.e(2)]).then(a.t.bind(null,411,7))]).then((function(a){var i=Object(r.a)(a,1)[0];i=i.default;var s=t.algoliaOptions,o=void 0===s?{}:s;i(Object.assign({},t,{inputSelector:"#algolia-search-input",algoliaOptions:Object.assign({facetFilters:["lang:".concat(e),"version:".concat(n.$site.base.split("/")[2])].concat(o.facetFilters||[])},o)}))}))},update:function(t,e){this.$el.innerHTML='<input id="algolia-search-input" class="search-query">',this.initialize(t,e)}}}),l=(a(365),Object(i.a)(o,(function(){var t=this.$createElement,e=this._self._c||t;return e("form",{staticClass:"algolia-search-wrapper search-box",attrs:{id:"search-form",role:"search"}},[e("input",{staticClass:"search-query",attrs:{id:"algolia-search-input",placeholder:this.placeholder}})])}),[],!1,null,null,null).exports),c=a(407),u=a(409),h=a(356);function p(t,e){return t.ownerDocument.defaultView.getComputedStyle(t,null)[e]}var d,v={name:"Navbar",components:{SidebarButton:u.a,NavLinks:h.a,SearchBox:c.a,AlgoliaSearchBox:l},data:function(){return{linksWrapMaxWidth:null}},computed:{algolia:function(){return this.$themeLocaleConfig.algolia||this.$site.themeConfig.algolia||{}},isAlgoliaSearch:function(){return this.algolia&&this.algolia.apiKey&&this.algolia.indexName}},mounted:function(){var t=this,e=parseInt(p(this.$el,"paddingLeft"))+parseInt(p(this.$el,"paddingRight")),a=function(){document.documentElement.clientWidth<719?t.linksWrapMaxWidth=null:t.linksWrapMaxWidth=t.$el.offsetWidth-e-(t.$refs.siteName&&t.$refs.siteName.offsetWidth||0)};a(),window.addEventListener("resize",a,!1)}},g=(a(373),Object(i.a)(v,(function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("header",{staticClass:"navbar"},[a("SidebarButton",{on:{"toggle-sidebar":function(e){return t.$emit("toggle-sidebar")}}}),t._v(" "),a("RouterLink",{staticClass:"home-link",attrs:{to:t.$localePath}},[t.$site.themeConfig.logo?a("img",{staticClass:"logo",attrs:{src:t.$withBase(t.$site.themeConfig.logo),alt:t.$siteTitle}}):t._e(),t._v(" "),t.$siteTitle?a("span",{ref:"siteName",staticClass:"site-name",class:{"can-hide":t.$site.themeConfig.logo}},[t._v("biome"),a("span",[t._v(".text")])]):t._e()]),t._v(" "),a("div",{staticClass:"links",style:t.linksWrapMaxWidth?{"max-width":t.linksWrapMaxWidth+"px"}:{}},[t.isAlgoliaSearch?a("AlgoliaSearchBox",{attrs:{options:t.algolia}}):!1!==t.$site.themeConfig.search&&!1!==t.$page.frontmatter.search?a("SearchBox"):t._e(),t._v(" "),a("NavLinks",{staticClass:"can-hide"})],1)],1)}),[],!1,null,null,null).exports),f=a(406),_=a(355),m=a(66),b=(a(104),a(70),a(27),a(321),a(171),a(49),a(381)),C=a.n(b),k={name:"Versions",data:function(){return{selected:void 0,options:[],showOptions:!1}},created:(d=Object(m.a)(regeneratorRuntime.mark((function t(){var e;return regeneratorRuntime.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return t.prev=0,t.next=3,C.a.get("https://raw.githubusercontent.com/recognai/biome-text/gh-pages/versions.txt");case 3:e=t.sent,this.options=e.data.split("\n").filter((function(t){return""!==t})).map((function(t){return t.trim()})),this.selected=window.location.pathname.split("/")[2],t.next=10;break;case 8:t.prev=8,t.t0=t.catch(0);case 10:case"end":return t.stop()}}),t,this,[[0,8]])}))),function(){return d.apply(this,arguments)}),methods:{onChange:function(t){this.showOptions=!1,this.selected=t;var e="/".concat(this.selected,"/"),a=window.location.pathname.split("/");window.location.pathname=a.slice(0,2).join("/")+e+a.slice(3).join("/")},clickOutside:function(){this.showOptions=!1}}},x=(a(399),Object(i.a)(k,(function(){var t=this,e=t.$createElement,a=t._self._c||e;return t.options&&t.options.length>0?a("span",{staticClass:"nav-versions"},[a("div",{staticClass:"nav-versions__select",on:{click:function(e){t.showOptions=!0}},model:{value:t.selected,callback:function(e){t.selected=e},expression:"selected"}},[a("strong",[t._v(t._s(t.selected))])]),t._v(" "),t.showOptions?a("div",{staticClass:"nav-versions__options__container"},[a("ul",{directives:[{name:"click-outside",rawName:"v-click-outside",value:t.clickOutside,expression:"clickOutside"}],staticClass:"nav-versions__options"},t._l(t.options,(function(e){return a("li",{staticClass:"nav-versions__option",attrs:{value:e},on:{click:function(a){return t.onChange(e)}}},[a("a",{class:e===t.selected?"active":"",attrs:{href:"#"}},[t._v(t._s(e))])])})),0)]):t._e()]):t._e()}),[],!1,null,null,null).exports),$={name:"Sidebar",components:{SidebarLinks:_.default,NavLinks:h.a,Versions:x},props:["items"]},w=(a(400),Object(i.a)($,(function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("aside",{staticClass:"sidebar"},[a("div",{staticClass:"sidebar__link"},[a("a",{attrs:{href:t.$withBase("/")}},[a("img",{staticClass:"sidebar__img",attrs:{src:t.$withBase("/assets/img/biome.svg")}})])]),t._v(" "),a("Versions"),t._v(" "),a("NavLinks"),t._v(" "),t._t("top"),t._v(" "),a("SidebarLinks",{attrs:{depth:0,items:t.items}}),t._v(" "),t._t("bottom")],2)}),[],!1,null,null,null).exports),S=a(318),O={name:"Layout",components:{Home:s,Page:f.a,Sidebar:w,Navbar:g},data:function(){return{isSidebarOpen:!1}},computed:{shouldShowNavbar:function(){var t=this.$site.themeConfig;return!1!==this.$page.frontmatter.navbar&&!1!==t.navbar&&(this.$title||t.logo||t.repo||t.nav||this.$themeLocaleConfig.nav)},shouldShowSidebar:function(){var t=this.$page.frontmatter;return!t.home&&!1!==t.sidebar&&this.sidebarItems.length},sidebarItems:function(){return Object(S.l)(this.$page,this.$page.regularPath,this.$site,this.$localePath)},pageClasses:function(){var t=this.$page.frontmatter.pageClass;return[{"no-navbar":!this.shouldShowNavbar,"sidebar-open":this.isSidebarOpen,"no-sidebar":!this.shouldShowSidebar},t]}},mounted:function(){var t=this;this.$router.afterEach((function(){t.isSidebarOpen=!1}))},methods:{toggleSidebar:function(t){this.isSidebarOpen="boolean"==typeof t?t:!this.isSidebarOpen,this.$emit("toggle-sidebar",this.isSidebarOpen)},onTouchStart:function(t){this.touchStart={x:t.changedTouches[0].clientX,y:t.changedTouches[0].clientY}},onTouchEnd:function(t){var e=t.changedTouches[0].clientX-this.touchStart.x,a=t.changedTouches[0].clientY-this.touchStart.y;Math.abs(e)>Math.abs(a)&&Math.abs(e)>40&&(e>0&&this.touchStart.x<=80?this.toggleSidebar(!0):this.toggleSidebar(!1))}}},L=(a(401),Object(i.a)(O,(function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"theme-container",class:t.pageClasses,on:{touchstart:t.onTouchStart,touchend:t.onTouchEnd}},[t.shouldShowNavbar?a("Navbar",{on:{"toggle-sidebar":t.toggleSidebar}}):t._e(),t._v(" "),a("div",{staticClass:"sidebar-mask",on:{click:function(e){return t.toggleSidebar(!1)}}}),t._v(" "),a("Sidebar",{attrs:{items:t.sidebarItems},on:{"toggle-sidebar":t.toggleSidebar},scopedSlots:t._u([{key:"top",fn:function(){return[t._t("sidebar-top")]},proxy:!0},{key:"bottom",fn:function(){return[t._t("sidebar-bottom")]},proxy:!0}],null,!0)}),t._v(" "),t.$page.frontmatter.home?a("Home"):a("Page",{attrs:{"sidebar-items":t.sidebarItems},scopedSlots:t._u([{key:"top",fn:function(){return[t._t("page-top")]},proxy:!0},{key:"bottom",fn:function(){return[a("footer",{staticClass:"footer"},[a("div",[t._v("\n          Maintained by\n          "),a("a",{attrs:{href:"https://www.recogn.ai/",target:"_blank"}},[a("img",{staticClass:"footer__img",attrs:{width:"70px",src:t.$withBase("/assets/img/recognai.png")}})])])]),t._v(" "),t._t("page-bottom")]},proxy:!0}],null,!0)})],1)}),[],!1,null,"348088ed",null));e.default=L.exports}}]);