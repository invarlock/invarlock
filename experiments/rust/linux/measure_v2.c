/* Linux process measurement v2: fresh exec image, event-driven child wait.
 * Do not reset peaks. The measured interval starts immediately before fork and
 * ends immediately after wait4 of the direct child. wait4 RSS includes its
 * pre-exec image (this small supervisor), not the outer Python address space.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/signalfd.h>
#include <sys/timerfd.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static void fail(const char *what) { perror(what); exit(125); }
static double seconds(struct timespec t) { return t.tv_sec + t.tv_nsec / 1e9; }
int main(int argc, char **argv) {
    if (argc < 5 || strcmp(argv[3], "--")) {
        fprintf(stderr, "usage: measure_v2 METRICS TIMEOUT -- COMMAND [ARG...]\n");
        return 125;
    }
    char *end;
    double timeout = strtod(argv[2], &end);
    if (*end || !isfinite(timeout) || timeout <= 0 || timeout > 86400) return 125;
    int result_fd = open(argv[1], O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (result_fd < 0) fail("open metrics");
    sigset_t block, old;
    sigemptyset(&block); sigaddset(&block, SIGCHLD);
    if (sigprocmask(SIG_BLOCK, &block, &old)) fail("sigprocmask");
    int events = signalfd(-1, &block, SFD_CLOEXEC);
    int timer = timerfd_create(CLOCK_MONOTONIC, TFD_CLOEXEC);
    int errors[2];
    if (events < 0 || timer < 0 || pipe2(errors, O_CLOEXEC)) fail("event setup");
    if (prctl(PR_SET_CHILD_SUBREAPER, 1)) fail("subreaper");
    struct timespec started, ended;
    if (clock_gettime(CLOCK_MONOTONIC, &started)) fail("clock");
    struct itimerspec limit = {0};
    double deadline = seconds(started) + timeout;
    limit.it_value.tv_sec = (time_t)deadline;
    limit.it_value.tv_nsec = (long)((deadline - limit.it_value.tv_sec) * 1e9);
    if (timerfd_settime(timer, TFD_TIMER_ABSTIME, &limit, NULL)) fail("timer arm");
    pid_t child = fork();
    if (child < 0) fail("fork");
    if (!child) {
        close(errors[0]);
        if (setpgid(0, 0) || sigprocmask(SIG_SETMASK, &old, NULL)) {
            int e = errno; (void)!write(errors[1], &e, sizeof e); _exit(127);
        }
        execvp(argv[4], &argv[4]);
        int e = errno; (void)!write(errors[1], &e, sizeof e); _exit(127);
    }
    close(errors[1]);
    /* Both sides setpgid: successful exec can race with this call. */
    if (setpgid(child, child) && errno != EACCES && errno != ESRCH) {
        kill(child, SIGKILL); waitpid(child, NULL, 0); fail("setpgid");
    }
    int timed_out = 0;
    struct pollfd fds[2] = {{events, POLLIN, 0}, {timer, POLLIN, 0}};
    for (;;) {
        siginfo_t info = {0};
        /* WNOWAIT pins the leader PID until group cleanup; never signal a
         * numeric group after reaping its leader (PID reuse hazard). */
        if (waitid(P_PID, child, &info, WEXITED | WNOHANG | WNOWAIT)) fail("waitid");
        if (info.si_pid) break;
        int ready = poll(fds, 2, -1);
        if (ready < 0) { if (errno == EINTR) continue; fail("poll"); }
        if (fds[0].revents & POLLIN) {
            struct signalfd_siginfo event;
            if (read(events, &event, sizeof event) != sizeof event) fail("signalfd read");
        }
        if (fds[1].revents & POLLIN) {
            /* Event ordering at the deadline: an already exited child wins. */
            memset(&info, 0, sizeof info);
            if (waitid(P_PID, child, &info, WEXITED | WNOHANG | WNOWAIT)) fail("deadline waitid");
            timed_out = !info.si_pid;
            break;
        }
    }
    /* Also clean up same-group descendants after normal direct-child exit.
     * The unreaped direct child pins the process-group identifier. */
    if (kill(-child, SIGKILL) && errno != ESRCH) fail("kill process group");
    struct rusage usage;
    int status;
    while (wait4(child, &status, 0, &usage) < 0) { if (errno != EINTR) fail("wait4"); }
    if (clock_gettime(CLOCK_MONOTONIC, &ended)) fail("end clock");
    /* Reap adopted same-group descendants, outside the measured interval. */
    while (waitpid(-child, NULL, 0) >= 0 || errno == EINTR) {}
    int exec_errno = 0;
    ssize_t count = read(errors[0], &exec_errno, sizeof exec_errno);
    if (count != 0 && count != sizeof exec_errno) fail("exec error pipe");
    int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -WTERMSIG(status);
    FILE *result = fdopen(result_fd, "w");
    if (!result) fail("fdopen");
    fprintf(result, "{\"format\":\"invarlock/linux-process-measurement-v2\","
        "\"exit_code\":%d,\"wait_status\":%d,\"timed_out\":%s,\"exec_errno\":%d,"
        "\"wall_seconds\":%.9f,\"user_cpu_seconds\":%.9f,\"system_cpu_seconds\":%.9f,"
        "\"max_rss_raw\":%ld,\"max_rss_raw_unit\":\"KiB\",\"max_rss_bytes\":%lld}\n",
        exit_code, status, timed_out ? "true" : "false", exec_errno,
        seconds(ended)-seconds(started),
        usage.ru_utime.tv_sec + usage.ru_utime.tv_usec/1e6,
        usage.ru_stime.tv_sec + usage.ru_stime.tv_usec/1e6,
        usage.ru_maxrss, (long long)usage.ru_maxrss*1024);
    if (fclose(result)) fail("write metrics");
    return 0;
}
