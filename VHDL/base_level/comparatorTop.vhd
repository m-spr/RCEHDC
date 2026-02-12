LIBRARY IEEE;
    USE IEEE.STD_LOGIC_1164.ALL;
    USE IEEE.NUMERIC_STD.ALL;

ENTITY comparatorTop IS
    GENERIC (len : INTEGER := 8;  -- bit width out adder
             n   : INTEGER := 10; -- #Classes       ---- all 4s in this code can be replaced with LOG2(n)
             z   : INTEGER := 10; -- zeropadding to 2**
             lgn : INTEGER := 4); ---LOG2(n)
    PORT (
        clk, rst, run           : IN  STD_LOGIC;
        a                       : IN  STD_LOGIC_VECTOR(n * len - 1 DOWNTO 0); --- 16 = 2**4 ,,, 4 is LOG2(n)
        done, TLAST_S, TVALID_S : OUT STD_LOGIC;                              --- final result is ready 
        classIndex              : OUT STD_LOGIC_VECTOR(lgn - 1 DOWNTO 0);      --- only the index of class can be also the value!  As of now only support up to 16 classes so 4'bits 
    
        predictedClassScore     : OUT STD_LOGIC_VECTOR(len - 1 DOWNTO 0);
        groundTruthScore        : OUT STD_LOGIC_VECTOR(len - 1 DOWNTO 0);
        ground_truth            : IN integer
        );
END ENTITY comparatorTop;

ARCHITECTURE behavioral OF comparatorTop IS
    COMPONENT recMux IS
        GENERIC (n      : INTEGER := 3; -- MUX sel bit width (number of layer)
                 lenPop : INTEGER := 4); -- bit width out popCounters
        PORT (
            sel  : IN  STD_LOGIC_VECTOR(n DOWNTO 0); -- sel<='0'&&sel
            din  : IN  STD_LOGIC_VECTOR(((2 ** n) * lenPop) - 1 DOWNTO 0);
            dout : OUT STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0)
        );
    END COMPONENT recMux;
    COMPONENT comparator IS
        GENERIC (len : INTEGER := 8); -- bit width out adder
        PORT (
            a, b : IN  STD_LOGIC_VECTOR(len - 1 DOWNTO 0);
            gr   : OUT STD_LOGIC --- case of a >= b gr ='1' other wise 0
        );
    END COMPONENT comparator;
    COMPONENT confCompCtrl IS
        GENERIC (n   : INTEGER := 10; -- #Classes
                 lgn : INTEGER := 4);
        PORT (
            clk, rst                        : IN  STD_LOGIC;
            run                             : IN  STD_LOGIC;
            runOut, done, TLAST_S, TVALID_S : OUT STD_LOGIC;
            pointer                         : OUT STD_LOGIC_VECTOR(lgn - 1 DOWNTO 0) --- As of now only support up to 16 classes so 4'bits
        );
    END COMPONENT confCompCtrl;
    COMPONENT reg IS
        GENERIC (lenPop : INTEGER := 8); -- bit width out popCounters
        PORT (
            clk               : IN  STD_LOGIC;
            regUpdate, regrst : IN  STD_LOGIC;
            din               : IN  STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0);
            dout              : OUT STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0)
        );
    END COMPONENT reg;
    COMPONENT reg1 IS
        GENERIC (lenPop : INTEGER := 8); -- bit width out popCounters
        PORT (
            clk               : IN  STD_LOGIC;
            regUpdate, regrst : IN  STD_LOGIC;
            din               : IN  STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0);
            dout              : OUT STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0)
        );
    END COMPONENT reg1;

    SIGNAL muxSel                    : STD_LOGIC_VECTOR(lgn - 1 DOWNTO 0);
    SIGNAL muxSelClassIdx            : STD_LOGIC_VECTOR(lgn - 1 DOWNTO 0);
    SIGNAL classIndexI, classIndexI2 : STD_LOGIC_VECTOR(lgn - 1 DOWNTO 0);
    SIGNAL muxSelE                   : STD_LOGIC_VECTOR(lgn DOWNTO 0);
    SIGNAL muxOut, toComp, fromComp  : STD_LOGIC_VECTOR(len - 1 DOWNTO 0);
    SIGNAL muxIn                     : STD_LOGIC_VECTOR(((2 ** lgn) + 1) * len - 1 DOWNTO 0);
    CONSTANT zero_muxin : STD_LOGIC_VECTOR(((z + 1) * len) - 1 DOWNTO 0) := (OTHERS => '0');
    SIGNAL regIndexupd, regIndexup, regUpdate, regrst, doneI : STD_LOGIC;

    ATTRIBUTE MARK_DEBUG                 : string;
    ATTRIBUTE MARK_DEBUG OF regUpdate    : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF muxSel       : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF toComp       : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF classIndexI2 : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF muxOut       : SIGNAL IS "TRUE";
BEGIN
    muxIn <= a & zero_muxin; ------- check!!!

    ctrl: confCompCtrl
        GENERIC MAP (n, lgn)
        PORT MAP (
            clk, rst, run,
            regrst, doneI, TLAST_S, TVALID_S, muxSel
        );

    CompMux: recMux
        GENERIC MAP (lgn,
                     len)
        PORT MAP (
            muxSelE, muxIn(((2 ** lgn) + 1) * len - 1 DOWNTO ((2 ** lgn) + 1) * len - ((2 ** lgn) * len)), muxOut ------- check!!!
        );

    regMax: reg1
        GENERIC MAP (len)
        PORT MAP (clk, regUpdate, regrst, muxOut, toComp);

    regIndexI: reg
        GENERIC MAP (lgn)
        PORT MAP (clk, regUpdate, rst, muxSel, classIndexI);

    regIndex: reg
        GENERIC MAP (lgn)
        PORT MAP (clk, doneI, rst, classIndexI2, classIndex);

    comp: comparator
        GENERIC MAP (len)
        PORT MAP (
            toComp, muxOut,
            regUpdate --- case of a >= b gr ='1' other wise 0
        );

    done         <= doneI;
    muxSelE      <= '0' & muxSel;
    classIndexI2 <= std_logic_vector(unsigned(n - 1 - unsigned(classIndexI)));
    muxSelClassIdx <= std_logic_vector(unsigned(n - 1 - unsigned(muxSel)));

    process(clk)
    begin
        if(to_integer(unsigned(muxSelClassIdx)) = ground_truth) then
            groundTruthScore <= muxOut;
        end if;
    end process;

    process(doneI)
    begin
        IF doneI = '1' THEN
            predictedClassScore <= toComp;
        END IF;
    end process;
    
END ARCHITECTURE behavioral;
