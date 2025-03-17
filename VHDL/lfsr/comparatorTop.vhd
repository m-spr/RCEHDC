-- MIT License

-- Copyright (c) 2024 m-spr

-- Permission is hereby granted, free of charge, to any person obtaining a copy
-- of this software and associated documentation files (the "Software"), to deal
-- in the Software without restriction, including without limitation the rights
-- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
-- copies of the Software, and to permit persons to whom the Software is
-- furnished to do so, subject to the following conditions:

-- The above copyright notice and this permission notice shall be included in all
-- copies or substantial portions of the Software.

-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
-- SOFTWARE.

LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

ENTITY comparatorTop IS
    GENERIC (
        len  : INTEGER := 8;  -- Bit width of adder output
        n    : INTEGER := 10; -- Number of classes (log2(n) can replace hardcoded values)
        z    : INTEGER := 10; -- Zero-padding to 2**
        lgn  : INTEGER := 4   -- Log2(n) equivalent for class selection
    );
    PORT (
        clk, rst, run : IN  STD_LOGIC;
        a             : IN  STD_LOGIC_VECTOR(n * len - 1 DOWNTO 0); 
        done, TLAST_S, TVALID_S : OUT STD_LOGIC;
        classIndex     : OUT STD_LOGIC_VECTOR(lgn - 1 DOWNTO 0)
    );
END ENTITY comparatorTop;

ARCHITECTURE behavioral OF comparatorTop IS

    -- Component Declarations
    COMPONENT recMux IS
        GENERIC (
            n      : INTEGER := 3;   -- MUX select bit width (number of layers)
            lenPop : INTEGER := 4    -- Bit width out popCounters
        );
        PORT (
            sel  : IN  STD_LOGIC_VECTOR(n DOWNTO 0);
            din  : IN  STD_LOGIC_VECTOR(((2**n) * lenPop) - 1 DOWNTO 0);
            dout : OUT STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0)
        ); 
    END COMPONENT;

    COMPONENT comparator IS
        GENERIC (
            len : INTEGER := 8  -- Bit width of comparator inputs
        );
        PORT (
            a, b : IN  STD_LOGIC_VECTOR(len - 1 DOWNTO 0);
            gr   : OUT STD_LOGIC  -- '1' when a >= b, otherwise '0'
        );
    END COMPONENT;

    COMPONENT confCompCtrl IS
        GENERIC (
            n   : INTEGER := 10;  -- Number of classes
            lgn : INTEGER := 4    -- Log2(n) for class indexing
        );
        PORT (
            clk, rst  : IN  STD_LOGIC;
            run       : IN  STD_LOGIC;
            runOut    : OUT STD_LOGIC;
            done      : OUT STD_LOGIC;
            TLAST_S   : OUT STD_LOGIC;
            TVALID_S  : OUT STD_LOGIC;
            pointer   : OUT STD_LOGIC_VECTOR(lgn - 1 DOWNTO 0)
        );
    END COMPONENT;

    COMPONENT reg IS
        GENERIC (
            lenPop : INTEGER := 8  -- Bit width of output popCounters
        );
        PORT (
            clk       : IN  STD_LOGIC;
            regUpdate : IN  STD_LOGIC;
            regrst    : IN  STD_LOGIC;
            din       : IN  STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0);
            dout      : OUT STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0)
        );
    END COMPONENT;

    COMPONENT reg1 IS
        GENERIC (
            lenPop : INTEGER := 8  -- Bit width of output popCounters
        );
        PORT (
            clk       : IN  STD_LOGIC;
            regUpdate : IN  STD_LOGIC;
            regrst    : IN  STD_LOGIC;
            din       : IN  STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0);
            dout      : OUT STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0)
        );
    END COMPONENT;

    -- Internal Signals
    SIGNAL muxSel      : STD_LOGIC_VECTOR(lgn - 1 DOWNTO 0);
    SIGNAL classIndexI, classIndexI2 : STD_LOGIC_VECTOR(lgn - 1 DOWNTO 0);
    SIGNAL muxSelE     : STD_LOGIC_VECTOR(lgn DOWNTO 0);
    SIGNAL muxOut, toComp, fromComp  : STD_LOGIC_VECTOR(len - 1 DOWNTO 0);
    SIGNAL muxIn       : STD_LOGIC_VECTOR(((2**lgn) + 1) * len - 1 DOWNTO 0);
    CONSTANT zero_muxin : STD_LOGIC_VECTOR(((z + 1) * len) - 1 DOWNTO 0) := (OTHERS => '0');
    SIGNAL regIndexupd, regIndexup, regUpdate, regrst, doneI : STD_LOGIC;

BEGIN

    -- Assign input with zero padding
    muxIn <= a & zero_muxin;  

    -- Control Unit Instance
    ctrl : confCompCtrl 
        GENERIC MAP (
            n   => n,
            lgn => lgn
        )
        PORT MAP (
            clk     => clk,
            rst     => rst,
            run     => run,
            runOut  => regrst,
            done    => doneI,
            TLAST_S => TLAST_S,
            TVALID_S => TVALID_S,
            pointer => muxSel
        );

    -- Recursive MUX Instance
    CompMux : recMux 
        GENERIC MAP (
            n      => lgn,
            lenPop => len
        )
        PORT MAP (
            sel  => muxSelE,
            din  => muxIn(((2**lgn) + 1) * len - 1 DOWNTO ((2**lgn) + 1) * len - ((2**lgn) * len)), 
            dout => muxOut  
        );

    -- Register for Maximum Value
    regMax : reg1 
        GENERIC MAP (
            lenPop => len
        )
        PORT MAP (
            clk       => clk,
            regUpdate => regUpdate,
            regrst    => regrst,
            din       => muxOut,
            dout      => toComp
        );

    -- Register for Class Index I
    regIndexI : reg 
        GENERIC MAP (
            lenPop => lgn
        )
        PORT MAP (
            clk       => clk,
            regUpdate => regUpdate,
            regrst    => rst,
            din       => muxSel,
            dout      => classIndexI
        );

    -- Register for Class Index Output
    regIndex : reg 
        GENERIC MAP (
            lenPop => lgn
        )
        PORT MAP (
            clk       => clk,
            regUpdate => doneI,
            regrst    => rst,
            din       => classIndexI2,
            dout      => classIndex
        );

    -- Comparator Instance
    comp : comparator 
        GENERIC MAP (
            len => len
        )
        PORT MAP (
            a  => toComp,
            b  => muxOut,
            gr => regUpdate   -- Case of a >= b gr = '1' otherwise 0
        );

    -- Output Assignments
    done     <= doneI;
    muxSelE  <= '0' & muxSel;
    classIndexI2 <= STD_LOGIC_VECTOR(UNSIGNED(n - 1 - UNSIGNED(classIndexI)));

END ARCHITECTURE behavioral;

