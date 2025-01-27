from tcvc import TCVC

tcvc = TCVC(
    "../../Test Videos/No GT/",
    "../results/Test/",
    "../experiments/TCVC_IDC/models/80000_G.pth",
)

tcvc.colorize()
