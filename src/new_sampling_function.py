def deriv(f,x):
    # I am sure python has a native version of this but I was on the plane.
    # replace with more elegant function
    h=0.0001*x
    return(  ( f(x+h)-f(x-hx) )/(2*h)  ) 


def sample_distance(self, PID, energy):
    if PID == 22:
        def n_sigma(energy):
            return (self._NSigmaPP(Energy) + self._NSigmaComp(Energy))

    elif PID == 11:
        def n_sigma(energy):
            return (self._NSigmaBrem(Energy) + self._NSigmaMoller(Energy))

    elif PID == -11:
        def n_sigma(energy):
            return (self._NSigmaBrem(Energy) + self._NSigmaAnn(Energy) + self._NSigmaBhabha(Energy))


    z_travelled =0
    hard_scatter=False
    var_energy  = energy

    while hard_scatter == False:

        random_number =  np.random.uniform(0.0, 1.0)

        ## Use first derivative of n_sigma to estimate a good step-size
        ## in coordinate space
        delta_z  =  cmtom/(deriv(n_sigma, var_energy)/n_sigma(var_energy) * self._dEdx)

        mfp = cmtom/n_sigma(var_energy)

    
        # Test if hard scatter happened
        if random_number > np.exp( -delta_z/mfp):
            hard_scatter = True
            final_energy = var_energy
            break()
        # If no hard scatter propagate particle
        # and account for energy loss
        else:
            hard_scatter = False

            var_energy= energy - dEdx*delta_z
            z_travelled = z_travelled+delta_z


    mfp = cmtom/n_sigma(final_energy)
    distC = np.random.uniform(0.0, 1.0)
    dist = z_travelled + mfp*np.log(1.0/(1.0+(np.exp(-delta_z/mfp)-1)*distC))

    # I have designed this code to interface with the currently
    # written function, however it is likely more elegant to
    # just do the energy losses etc in this function, and return
    # the final "parent" kinematics 
    
    return(dist) 

        
        
