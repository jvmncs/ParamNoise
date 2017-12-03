def testDQN(env, model, args):
    model.eval()
    return env, args

def testPPO(env, model, args):
    model.eval()
    return env, args
