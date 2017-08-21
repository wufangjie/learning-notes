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
    (set-font "monaco" "hannotate sc" 18 22))


;; =====================================================================
;; `basic'
;; =====================================================================
(global-font-lock-mode t)
(show-paren-mode t)
(setq show-paren-style 'parentheses)

(setq inhibit-startup-message 0)
(setq frame-title-format "%b")
(tool-bar-mode 0)
(menu-bar-mode 0)
(scroll-bar-mode 0)

(global-linum-mode t)
(setq column-number-mode t)
(setq scroll-margin 3
      scroll-conservatively 10000)

(setq blink-cursor-mode nil)
; (setq blink-cursor-delay 1)

(set-frame-parameter (selected-frame) 'alpha '(95 95))

(setq x-select-enable-clipboard t)  ; shared with clipboard
(setq make-backup-files nil)        ; no ~ file
(setq ediff-split-window-function 'split-window-horizontally)
(setq org-src-fontify-natively t)   ; highlight the org-mode's source code

(add-hook 'before-save-hook 'delete-trailing-whitespace)


(mapc (lambda (mode)
        (font-lock-add-keywords
         mode
         '(("\\<\\(FIXME\\|TODO\\|NOTE\\)"
	    1 'font-lock-warning-face prepend))))
      '(python-mode org-mode emacs-lisp-mode c-mode))


(add-hook 'org-mode-hook
	  (lambda ()
	    (linum-mode -1)
	    ))


(unless window-system
  (defun clipboard-save ()
    (interactive)
    (call-process-region (point-min) (point-max)
			 "xsel" nil 0 nil "--clipboard" "--input")))


;; =====================================================================
;; `global-kbd'
;; =====================================================================
; use M-x suspend-frame instead of shortcuts, it's dangerous
(global-unset-key (kbd "C-z"))
(global-unset-key (kbd "C-x C-z"))

;; ;; If you use emacs -nw, M-[ kbd will drive you crazy
;; (global-unset-key (kbd "M-}"))
;; (global-unset-key (kbd "M-{"))
;; (global-set-key (kbd "M-]") 'forward-paragraph)
;; (global-set-key (kbd "M-[") 'backward-paragraph)

(if window-system
    (global-set-key (kbd "C-l")
		    (lambda ()
		      (interactive)
		      (recenter-top-bottom)
		      (redraw-display))))

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
(set-background-color "#131926")
(set-foreground-color "#ffffff")
;ShortcutsNoMnemonics=TRUE ;#98bff3
;; (:overline t) the heights not equal
;; (:box (:line-width -1)) not monospaced, not as elisp's doc say, it's a bug
;; :box (:line-width -1 :style "released-button")))) t)

(unless window-system
  ; black red green yellow blue magenta cyan white
  (custom-set-faces
   '(font-lock-builtin-face ((t (:foreground "magenta" :bold t))) t)
   '(font-lock-comment-face ((t (:foreground "green"))) t)
   '(font-lock-constant-face ((t (:foreground "yellow" :bold t))) t)
   '(font-lock-function-name-face ((t (:background "blue" :underline t))) t)
   '(font-lock-keyword-face ((t (:foreground "cyan" :bold t))) t)
   '(font-lock-string-face ((t (:foreground "yellow"))) t)
   '(font-lock-type-face ((t (:foreground "green" :bold t))) t)
   '(font-lock-variable-name-face ((t (:foreground "blue" :bold t))) t)
   '(font-lock-warning-face ((t (:foreground "magenta" :bold t))) t)

   '(link ((t (:inherit font-lock-function-name-face))) t)
   '(link-visited
     ((t (:foreground "black" :background "magenta" :underline t))) t)
   '(region ((t (:background "blue"))) t)
   '(shadow ((t (:foreground "cyan"))) t)
   '(error ((t (:foreground "red" :bold t))) t)
   '(cursor ((t (:background "white"))) t)
   '(org-block ((t (:foreground "white"))) t)
   '(org-table ((t (:foreground "blue"))) t)
   '(linum ((t (:foreground "black" :underline t))) t)
   '(isearch ((t (:foreground "white" :background "green"))) t)
   '(isearch-fail ((t (:foreground "white" :background "red"))) t)
   '(highlight ((t (:inherit isearch))) t)
   '(lazy-highlight ((t (:foreground "white" :background "blue"))) t)
   '(popup-face ((t (:foreground "black" :background "cyan"))) t)
   '(popup-summary-face ((t (:foreground "blue" :background "cyan"))) t)
   '(popup-tip-face ((t (:foreground "white" :background "blue"))) t)
   '(popup-menu-selection-face ((t (:inherit popup-tip-face))) t)
   '(popup-selection-face ((t (:inherit popup-tip-face))) t)

   '(holiday ((t (:inherit link))) t)
   '(diary ((t (:inherit link-visited))) t)
   '(button ((t (:inherit link))) t)
   '(minibuffer-prompt ((t (:inherit link))) t)
   '(eshell-prompt ((t (:inherit link))) t)
   '(show-paren-match ((t (:inherit isearch :bold t))) t)
   '(show-paren-mismatch ((t (:inherit isearch-fail :bold t))) t)
   '(warning ((t (:inherit font-lock-warning-face))) t)
   '(success ((t (:inherit font-lock-type-face))) t)
   ))

(if window-system
  ; black red green yellow blue magenta cyan white
  (custom-set-faces
   '(font-lock-builtin-face ((t (:foreground "#ffbbff" :bold t))) t)
   '(font-lock-comment-face ((t (:foreground "#66cd00"))) t)
   '(font-lock-constant-face ((t (:foreground "#ffb90f" :bold t))) t)
   '(font-lock-function-name-face ((t (:background "#4f94cd" :underline t))) t)
   '(font-lock-keyword-face ((t (:foreground "#00ffff" :bold t))) t)
   '(font-lock-string-face ((t (:foreground "#ffa07a"))) t)
   '(font-lock-type-face ((t (:foreground "#9aff9a" :bold t))) t)
   '(font-lock-variable-name-face ((t (:foreground "#ffec8b" :bold t))) t)
   '(font-lock-warning-face ((t (:foreground "#ffbbff" :bold t))) t)

   '(link ((t (:inherit font-lock-function-name-face))) t)
   '(link-visited
     ((t (:foreground "#000000" :background "#ffbbff" :underline t))) t)
   '(region ((t (:background "#4f94cd"))) t)
   '(shadow ((t (:foreground "#d3d3d3"))) t)
   '(error ((t (:foreground "#ff3030" :bold t))) t)
   '(cursor ((t (:background "#ffffff"))) t)
   '(org-block ((t (:foreground "#ffffff"))) t)
   '(org-table ((t (:foreground "#4f94cd"))) t)
   '(isearch ((t (:foreground "#ffffff" :background "#66cd00"))) t)
   '(isearch-fail ((t (:foreground "#ffffff" :background "#ff3030"))) t)
   '(highlight ((t (:inherit isearch))) t)
   '(lazy-highlight ((t (:foreground "#ffffff" :background "#4f94cd"))) t)
   '(popup-face ((t (:foreground "#000000" :background "#d3d3d3"))) t)
   '(popup-summary-face ((t (:foreground "#4f94cd" :background "#d3d3d3"))) t)
   '(popup-tip-face ((t (:foreground "#ffffff" :background "#4f94cd"))) t)
   '(popup-menu-selection-face ((t (:inherit popup-tip-face))) t)
   '(popup-selection-face ((t (:inherit popup-tip-face))) t)

   '(holiday ((t (:inherit link))) t)
   '(diary ((t (:inherit link-visited))) t)
   '(button ((t (:inherit link))) t)
   '(minibuffer-prompt ((t (:inherit link))) t)
   '(eshell-prompt ((t (:inherit link))) t)
   '(show-paren-match ((t (:inherit isearch :bold t))) t)
   '(show-paren-mismatch ((t (:inherit isearch-fail :bold t))) t)
   '(warning ((t (:inherit font-lock-warning-face))) t)
   '(success ((t (:inherit font-lock-type-face))) t)

   '(linum ((t (:foreground "#666666" :underline t))) t)
   '(ediff-even-diff-A ((t (:background "#666666"))) t)
   '(ediff-even-diff-Ancestor ((t (:background "#666666"))) t)
   '(ediff-even-diff-B ((t (:background "#666666"))) t)
   '(ediff-even-diff-C ((t (:background "#666666"))) t)
   '(ediff-odd-diff-A ((t (:background "#666666"))) t)
   '(ediff-odd-diff-Ancestor ((t (:background "#666666"))) t)
   '(ediff-odd-diff-B ((t (:background "#666666"))) t)
   '(ediff-odd-diff-C ((t (:background "#666666"))) t)
   ))





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

(add-hook 'calendar-today-visible-hook 'calendar-star-date)


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
(setq fast-search-hash-table #s(hash-table test equal))
(dolist
    (pair
     `(("baidu"         . "https://www.baidu.com/s?wd=%s")
       ("bing"          . "https://www.bing.com/search?q=%s")
       ("bing-global"   . ,(concat "https://global.bing.com/search?q=%s"
				  "&setmkt=en-us&setlang=en-us&FORM=SECNEN"))
       ("bing-dict"     . "https://www.bing.com/dict/search?q=%s")
       ("douban"        . "https://www.douban.com/search?q=%s")
       ("douban-book"   . ,(concat "https://book.douban.com/subject_search"
				  "?search_text=%s&cat=1001"))
       ("douban-movie"  . ,(concat "https://movie.douban.com/subject_search"
				  "?search_text=%s&cat=1002"))
       ("github"        . "https://github.com/search?q=%s")
       ("jingdong"      . "https://search.jd.com/Search?keyword=%s&enc=utf-8")
       ("stackoverflow" . "https://stackoverflow.com/search?q=%s")
       ("taobao"        . "https://s.taobao.com/search?q=%s")
       ("wiki"          . "https://en.wikipedia.org/wiki/%s")
       ("zhihu"         . "https://www.zhihu.com/search?type=content&q=%s")))
  (puthash (car pair) (cdr pair) fast-search-hash-table))

;; ^~-:.@%+=              ; do not change
;; []{}                   ; i think should change
;; $#                     ; emacs change them
;;  \t\n`<>|;!?'"()*\&    ; must change
(defun convert-to-bash-safe-file-name (name)
  (replace-regexp-in-string
   "\\([][ \t\n`<>|;!?'\"()*\\&$#{}]\\)" "\\\\\\1" name))
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
(setq python-template #s(hash-table test equal))
(dolist
    (pair
     `(("head" . "#!/usr/bin/python3\n# -*- coding: utf-8 -*-\n\n")
       ("main" . "if __name__ == '__main__':\n    ")
       ("path" . ,(concat
		   "import os\n\n\ntry:\n"
		   "    path = os.path.split(os.path.realpath(__file__))[0]\n"
		   "except NameError:\n"
		   "    path = os.getcwd() or os.getenv('PWD')\n\n"))
       ("pdb"  . "import pdb; pdb.set_trace()\n")))
  (puthash (car pair) (cdr pair) python-template))


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
;; `gnu/linux' `darwin' `windows-nt'
;; =====================================================================
(if (string-equal system-type "gnu/linux")
    (call-process-shell-command "xmodmap ~/.xmodmap") nil 0)


;; =====================================================================
;; load other files
;; =====================================================================
(load "~/.emacs.d/my-holidays.el")
