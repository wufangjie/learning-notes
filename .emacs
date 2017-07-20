;; =====================================================================
;; 等宽测试|
;; ABCDefgh|
;; =====================================================================
(defun set-font (english chinese english-size chinese-size)
  (set-face-attribute 'default nil :font
		      (format "%s:pixelsize=%d"
			      english
			      english-size))
  (dolist (charset '(kana han symbol cjk-misc bopomofo))
    (set-fontset-font (frame-parameter nil 'font)
		      charset
                      (font-spec :family chinese
				 :size chinese-size))))
(if window-system
    (set-font "monaco" "hannotate sc" 16 20))


;; =====================================================================
;; `basic'
;; =====================================================================
(global-font-lock-mode t)
(show-paren-mode t)
(setq show-paren-style 'parentheses)

(setq inhibit-startup-message 0)
(tool-bar-mode 0)
(menu-bar-mode 0)
(scroll-bar-mode 0)

(setq frame-title-format "%b")
(setq column-number-mode t)
(global-linum-mode t)
(setq scroll-margin 3
      scroll-conservatively 10000)

(setq blink-cursor-mode nil)
;(setq blink-cursor-delay 1)


(set-frame-parameter (selected-frame) 'alpha '(90 50))


(setq x-select-enable-clipboard t)  ; shared with clipboard
(setq make-backup-files nil)  ; no ~ file
(add-hook 'before-save-hook 'delete-trailing-whitespace)
(setq ediff-split-window-function 'split-window-horizontally)
(setq org-src-fontify-natively t)  ; highlight the org-mode's source code

(mapc (lambda (mode)
        (font-lock-add-keywords
         mode
         '(("\\<\\(FIXME\\|TODO\\|NOTE\\)"
	    1 'font-lock-warning-face prepend))))
      '(python-mode org-mode emacs-lisp-mode c-mode))


;; =====================================================================
;; `global-kbd'
;; =====================================================================
(global-unset-key (kbd "M-}"))
(global-unset-key (kbd "M-{"))
(global-set-key (kbd "M-]") 'forward-paragraph)
(global-set-key (kbd "M-[") 'backward-paragraph)

(global-set-key (kbd "C-l")
		(lambda ()
		  (interactive)
		  (recenter-top-bottom)
		  (redraw-display)))

(global-set-key (kbd "C-x C-b") 'ibuffer)

(add-hook 'ibuffer-mode-hook
	  (lambda ()
	    (local-set-key (kbd "U") 'ibuffer-unmark-all)
	    ))


;; =====================================================================
;; `python'
;; hide / show code, M-x hs-<TAB>
;; =====================================================================
(setq python-shell-interpreter "python3")
(add-hook 'python-mode-hook 'hs-minor-mode)
(add-hook 'inferior-python-mode-hook
	  (lambda ()
	    (outline-minor-mode t)
	    (setq outline-regexp "\\(>>> \\)+")
	    ))


;; =====================================================================
;; `color'
;; useful functions: list-color-display list-faces-display describe-face
;; =====================================================================
(set-background-color "#111111")
(set-foreground-color "#ffffff")
;(set-face-foreground 'region "#ffffff")
(set-face-background 'region "#228b22")


;; (:overline t) the heights not equal
;; (:box (:line-width -1)) not monospaced, not as elisp's doc say, it's a bug
;; :box (:line-width -1 :style "released-button")))) t)
(custom-set-faces
 '(font-lock-builtin-face ((t (:foreground "#ffbbff"))) t)
 '(font-lock-comment-face ((t (:foreground "#66cd00"))) t)
 '(font-lock-constant-face ((t (:foreground "#ffb90f" :weight bold))) t)
 '(font-lock-function-name-face ((t (:background "#4f94cd" :underline t))) t)
 '(font-lock-keyword-face ((t (:foreground "#00ffff" :weight bold))) t)
 '(font-lock-string-face ((t (:foreground "#ffa07a"))) t)
 '(font-lock-type-face ((t (:foreground "#9aff9a" :weight bold))) t)
 '(font-lock-variable-name-face ((t (:foreground "#ffec8b" :bold t))) t)
 '(holiday ((t (:background "#4f94cd" :underline t))) t)
 '(diary ((t (:foreground "#000000" :background "#ffbbff" :underline t))) t)
 '(linum ((t (:foreground "#333333" :underline t))) t)
 '(button ((t (:foreground "#4f94cd" :underline t))) t)
 '(highlight ((t (:foreground "#ffffff" :background "#66cd00"))) t)
 '(isearch ((t (:foreground "#4f94cd" :background "#ffffff" :underline t))) t)
 '(lazy-highlight ((t (:foreground "#ffffff" :background "#4f94cd"))) t)
 '(popup-face ((t (:foreground "#000000" :background "#d3d3d3"))) t)
 '(popup-tip-face ((t (:foreground "#ffffff" :background "#4f94cd"))) t)
 '(popup-summary-face ((t (:foreground "#4f94cd" :background "#d3d3d3"))) t)
 '(cursor ((t (:background "#ffffff"))) t)
 '(org-block ((t (:foreground "#ffffff"))) t)
 ;'(org-table ((t (:foreground "#"))) t)
 '(isearch-fail ((t (:foreground "#ffffff" :background "#ee0000"))) t)
 '(minibuffer-prompt ((t (:inherit font-lock-function-name-face))) t)
 '(show-paren-mismatch ((t (:inherit isearch-fail))) t)
 )


;; =====================================================================
;; `datetime'
;; =====================================================================
(setq display-time-24hr-format t)
(setq display-time-format "%H:%M:%S %m-%d %a")
(setq display-time-interval 1)
(display-time-mode t)

(add-hook 'calendar-mode-hook
	  (lambda ()
	    (local-set-key (kbd "q") 'kill-buffer-and-window)
	    (setq calendar-week-start-day 1)
	    (setq calendar-mark-holidays-flag t)
	    (setq diary-file "~/.emacs.d/my-diary.txt")
	    ))


;; =====================================================================
;; `jedi'
;; =====================================================================
(add-to-list 'load-path "~/.emacs.d/el-get/el-get")
(unless (require 'el-get nil 'noerror)
  (with-current-buffer
      (url-retrieve-synchronously
       "https://raw.github.com/dimitri/el-get/master/el-get-install.el")
    (goto-char (point-max))
    (eval-print-last-sexp)))
(el-get 'sync)

(setq jedi:server-args
      '("--sys-path" "/usr/lib/python3/dist-packages"
	"--sys-path" "/usr/local/lib/python3.5/dist-packages"
	"--sys-path" "/usr/lib/python3.5/dist-packages"
	"--sys-path" "/home/wfj/packages"))

(setq jedi:setup-keys t)
(setq jedi:complete-on-dot t)
(add-hook 'python-mode-hook 'jedi:setup)
(add-hook 'inferior-python-mode-hook 'jedi:setup)


;; =====================================================================
;; `eshell'
;; use M-[/] instead of outline-mode
;; =====================================================================
(setq eshell-save-history-on-exit t
      eshell-history-size 4096
      eshell-hist-ignoredups t)

;; (add-hook 'eshell-mode-hook
;; 	  (lambda ()
;; 	    (outline-minor-mode t)
;; 	    (setq outline-regexp "[~/][^\n]*? [#$] ")
;; 	    ))

(defvar eshell-histignore
  '("^\\(cd\\|git\\|svn\\|g\\+\\+\\|cc\\|nvcc\\)\\(\\(\\s \\)+.*\\)?$"
    "^("
    "^\\./a\\.out$"
    "^xmodmap ~/\\.xmodmap$"
    "^sudo apt-get \\(update\\|upgrade\\|autoremove\\)$"
    "^man "
    "^\\(sudo \\)?pip[23]? \\(list\\|show\\|search\\)"
    " -\\(-version\\|[Vv]\\)$"
    ))

(setq eshell-input-filter
      #'(lambda (str)
	  (let ((regex eshell-histignore))
	    (not
	     (catch 'break
	       (while regex
		 (if (string-match (pop regex) str)
		     (throw 'break t))))))))


;; =====================================================================
;; `c-style'
;; =====================================================================
(add-hook 'c-mode-hook
	  (lambda ()
	      (c-set-style "ellemtel")
	      (setq c-basic-offset 4)
	      (setq indent-tabs-mode nil)
              (local-set-key (kbd "C-c d")
                             (lambda ()
                               (interactive)
                               (manual-entry (current-word))))
	      ))


;; =====================================================================
;; fast search keywords on internet, fast open file
;; TODO? use`open' for mac, `start' for windows instead of `xdg-open'
;; =====================================================================
(setq fast-search-hash-table
      #s(hash-table
	 test equal
	 data ("baidu"         "https://www.baidu.com/s?wd=%s"
	       "bing"          "https://www.bing.com/search?q=%s"
	       "bing-global"   "https://global.bing.com/search?q=%s&setmkt=en-us&setlang=en-us&FORM=SECNEN"
	       "bing-dict"     "https://www.bing.com/dict/search?q=%s"
	       "douban"        "https://www.douban.com/search?q=%s"
	       "douban-book"   "https://book.douban.com/subject_search?search_text=%s&cat=1001"
	       "douban-movie"  "https://movie.douban.com/subject_search?search_text=%s&cat=1002"
	       "github"        "https://github.com/search?q=%s"
	       "jingdong"      "https://search.jd.com/Search?keyword=%s&enc=utf-8"
	       "stackoverflow" "https://stackoverflow.com/search?q=%s"
	       "taobao"        "https://s.taobao.com/search?q=%s"
	       "wiki"          "https://en.wikipedia.org/wiki/%s"
	       "zhihu"         "https://www.zhihu.com/search?type=content&q=%s"
	       )))

;; ^~-:.@%+=              ; do not change
;; []{}                   ; i think should change
;; $#                     ; emacs change them
;;  \t\n`<>|;!?'"()*\&    ; must change
(defun convert-to-bash-safe-file-name (name)
  (replace-regexp-in-string "\\([][ \t\n`<>|;!?'\"()*\\&$#{}]\\)" "\\\\\\1" name))
  ;(replace-regexp-in-string "\\([^a-zA-Z0-9]\\)" "\\\\\\1" name))

(defun fast-search ()
  (interactive)
  (let* ((type (completing-read "Type wanted: " fast-search-hash-table))
	 (url (gethash type fast-search-hash-table)))
    (if url
	(let ((name (read-from-minibuffer "Name wanted: " )))
	  (if name
	      (call-process-shell-command
	       (concat "xdg-open " (convert-to-bash-safe-file-name
				    (format url name))) nil 0)
	    (message "You did't specify a name")))
      (message "Unknown type"))))

(defun fast-open (name)
  (interactive
   (list (read-file-name "Open file: ")))
  (if (file-exists-p name)
      (call-process-shell-command
       (concat "xdg-open " (convert-to-bash-safe-file-name
			    (expand-file-name name))) nil 0)
    (message "Invalid file")))


;; =====================================================================
;; `dired-mode'
;; =====================================================================
(add-hook 'dired-mode-hook
	  (lambda ()
	    (setq dired-actual-switches "-la")
	    (setq dired-recursive-copies "always")
	    (setq dired-recursive-deletes "always")
	    (local-set-key (kbd "RET")
			   (lambda ()
			     (interactive)
			     (let ((name (dired-file-name-at-point)))
			       (if (file-directory-p name)
				   (dired-find-file)
				 (fast-open name)))))
	    (local-set-key (kbd "s")
			   (lambda ()
			     (interactive)
			     (when dired-sort-inhibit
			       (error "Cannot sort this Dired buffer"))
			     (dired-sort-other
			      (read-string "ls switches (must contain -l): "
					   dired-actual-switches))))
	    ))


;; =====================================================================
;; `insert-python-template'
;; =====================================================================
(setq python-template
      #s(hash-table
	 test equal
	 data ("head"    "#!/usr/bin/python3\n# -*- coding: utf-8 -*-\n\n"
	       "main"    "if __name__ == '__main__':\n    "
	       "path"    "import os\n\n\ntry:\n    path = os.path.split(os.path.realpath(__file__))[0]\nexcept NameError:\n    path = os.getcwd() or os.getenv('PWD')\n\n"
	       )))

(defun insert-python-template ()
  (interactive)
  (let* ((type (completing-read "Type wanted: " python-template))
	 (content (gethash type python-template "")))
    (princ content (current-buffer))))


;; =====================================================================
;; `truncate-lines'
;; =====================================================================
(add-hook 'org-mode-hook (lambda () (setq truncate-lines nil)))
;; (add-hook 'eshell-mode-hook (lambda () (setq truncate-lines t)))
(add-hook 'sql-interactive-mode-hook (lambda () (setq truncate-lines t)))





;; =====================================================================
;; load other files
;; =====================================================================
(load "~/.emacs.d/my-holidays.el")




; TODO: hash-talbe's string is too long a line, make it short
